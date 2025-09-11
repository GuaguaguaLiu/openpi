"""OpenPi 模型训练脚本。

该脚本实现了完整的模型训练流程，包括：
- 模型初始化和权重加载
- 数据加载和预处理
- 训练循环和优化
- 检查点保存和恢复
- 实验日志记录（WandB）

使用方法：
    python scripts/train.py <config_name> --exp_name <experiment_name>

例如：
    python scripts/train.py pi0_libero --exp_name my_experiment
"""

import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """初始化自定义日志格式以提高可读性。

    设置日志级别映射和自定义格式化器，使日志输出更简洁易读。
    """
    # 日志级别映射：将完整级别名映射为单字母缩写
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        """自定义日志格式化器，使用简化的级别名称。"""
        
        def format(self, record):
            """格式化日志记录，使用简化的级别名称。"""
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    # 创建自定义格式化器
    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    # 配置根日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    """初始化 WandB 实验跟踪。

    Args:
        config: 训练配置对象。
        resuming: 是否从检查点恢复训练。
        log_code: 是否记录代码到 WandB。
        enabled: 是否启用 WandB 日志记录。

    Raises:
        FileNotFoundError: 如果检查点目录不存在。
    """
    if not enabled:
        # 禁用 WandB 模式
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    
    if resuming:
        # 恢复训练：从检查点目录读取 WandB 运行 ID
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        # 新训练：创建新的 WandB 运行
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        # 保存 WandB 运行 ID 到检查点目录
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        # 记录项目代码到 WandB
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """加载并验证权重，返回加载的权重子集。

    Args:
        loader: 权重加载器。
        params_shape: 参数形状结构。

    Returns:
        加载并验证后的参数。
    """
    loaded_params = loader.load(params_shape)
    # 验证加载的参数形状和数据类型
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # 从加载的参数中移除 jax.ShapeDtypeStruct，确保只返回实际加载的参数
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    """初始化训练状态。

    Args:
        config: 训练配置对象。
        init_rng: 初始化随机数生成器。
        mesh: JAX 分片网格。
        resume: 是否从检查点恢复训练。

    Returns:
        训练状态和分片信息的元组。
    """
    # 创建优化器
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        """初始化训练状态的内部函数。

        Args:
            rng: 随机数生成器。
            partial_params: 部分预训练参数。

        Returns:
            初始化的训练状态。
        """
        rng, model_rng = jax.random.split(rng)
        # 初始化模型（及其参数）
        model = config.model.create(model_rng)

        # 将部分参数合并到模型中
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # 如果部分参数不是状态的子集，这将产生错误
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # 将冻结参数转换为 bfloat16 精度
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    # 评估训练状态形状
    train_state_shape = jax.eval_shape(init, init_rng)
    # 创建 FSDP 分片
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        # 恢复训练时只返回形状和分片信息
        return train_state_shape, state_sharding

    # 加载并验证预训练权重
    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # 初始化训练状态并合并部分参数
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # 捐赠部分参数缓冲区
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """执行一个训练步骤。

    Args:
        config: 训练配置对象。
        rng: 随机数生成器。
        state: 当前训练状态。
        batch: 训练批次数据（观测和动作）。

    Returns:
        更新后的训练状态和训练信息的元组。
    """
    # 合并模型定义和参数
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        """计算损失函数。

        Args:
            model: 模型实例。
            rng: 随机数生成器。
            observation: 观测数据。
            actions: 动作数据。

        Returns:
            平均损失值。
        """
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    # 为当前步骤生成随机数
    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # 过滤掉冻结的参数
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    # 获取可训练参数并应用优化器更新
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # 就地更新模型并返回新的完整状态
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    # 创建新的训练状态
    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    
    # 更新指数移动平均参数
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # 过滤出非核参数（用于参数范数计算）
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    
    # 收集训练信息
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def main(config: _config.TrainConfig):
    """主训练函数。

    Args:
        config: 训练配置对象。
    """
    # 初始化日志记录
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    # 验证批次大小与设备数量的兼容性
    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    # 配置 JAX 编译缓存目录
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    # 初始化随机数生成器
    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    # 创建分片网格和分片策略
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # 初始化检查点管理器
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )

    # 初始化 WandB 实验跟踪
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # 创建数据加载器
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # 记录第一批次的图像用于检查
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    # 初始化训练状态
    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    # 如果恢复训练，从检查点恢复状态
    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    # 编译训练步骤函数
    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    # 设置训练循环
    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    # 训练循环
    infos = []
    for step in pbar:
        # 执行训练步骤
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        
        # 记录训练指标
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        
        # 获取下一个批次
        batch = next(data_iter)

        # 保存检查点
        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    # 等待检查点管理器完成
    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    # 使用配置命令行接口启动训练
    # cli() 会创建命令行接口, 解析命令行参数 替换 TrainConfig 内容，然后返回 TrainConfig 对象给main 函数
    main(_config.cli())
