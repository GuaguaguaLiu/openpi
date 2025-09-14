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

    这个函数设置日志级别映射和自定义格式化器，使日志输出更简洁易读。
    
    主要功能：
    1. 将日志级别名简化为单字母（DEBUG→D, INFO→I等）
    2. 自定义日志格式，包含时间、级别、消息、进程ID、文件名、行号
    3. 配置根日志记录器使用新的格式化器
    """
    # ========== 第一步：设置日志级别映射 ==========
    # 将完整级别名映射为单字母缩写，使日志输出更紧凑
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    # ========== 第二步：定义自定义格式化器类 ==========
    class CustomFormatter(logging.Formatter):
        """自定义日志格式化器，使用简化的级别名称。"""
        
        def format(self, record):
            """
            格式化日志记录，使用简化的级别名称。
            
            Args:
                record: 日志记录对象
                
            Returns:
                格式化后的日志字符串
            """
            # 将完整级别名替换为单字母缩写
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    # ========== 第三步：创建自定义格式化器 ==========
    # 定义日志格式：时间.毫秒 [级别] 消息(进程ID:文件名:行号)
    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",  # 时间格式：小时:分钟:秒
    )

    # ========== 第四步：配置根日志记录器 ==========
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # 设置日志级别为INFO
    logger.handlers[0].setFormatter(formatter)  # 应用自定义格式化器


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    """初始化 WandB 实验跟踪。

    这个函数负责初始化Weights & Biases (WandB)实验跟踪系统，用于记录训练指标、配置参数等。
    
    主要功能：
    1. 根据配置决定是否启用WandB
    2. 处理恢复训练和新训练两种情况
    3. 保存和恢复WandB运行ID
    4. 可选地记录项目代码

    Args:
        config: 训练配置对象，包含项目名称、实验名称等
        resuming: 是否从检查点恢复训练
        log_code: 是否记录代码到WandB
        enabled: 是否启用WandB日志记录

    Raises:
        FileNotFoundError: 如果检查点目录不存在
    """
    # ========== 第一步：检查是否启用WandB ==========
    if not enabled:
        # 禁用WandB模式：初始化但不记录任何数据
        wandb.init(mode="disabled")
        return

    # ========== 第二步：验证检查点目录 ==========
    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    
    # ========== 第三步：根据训练类型初始化WandB ==========
    if resuming:
        # 恢复训练：从检查点目录读取WandB运行ID并恢复会话
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        # 新训练：创建新的WandB运行
        wandb.init(
            name=config.exp_name,                    # 实验名称
            config=dataclasses.asdict(config),       # 将配置对象转换为字典
            project=config.project_name,             # 项目名称
        )
        # 保存WandB运行ID到检查点目录，用于后续恢复
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    # ========== 第四步：可选地记录项目代码 ==========
    if log_code:
        # 记录项目代码到WandB，便于实验复现
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """加载并验证权重，返回加载的权重子集。

    这个函数负责从权重加载器加载预训练权重，并验证其形状和数据类型。
    
    主要功能：
    1. 使用权重加载器加载预训练权重
    2. 验证加载的权重形状和数据类型是否匹配
    3. 过滤掉未加载的参数（ShapeDtypeStruct），只返回实际加载的权重

    Args:
        loader: 权重加载器，负责从磁盘或网络加载权重
        params_shape: 参数形状结构，定义期望的参数格式

    Returns:
        加载并验证后的参数，只包含实际加载的权重
        
    Raises:
        ValueError: 如果加载的权重形状或类型不匹配
    """
    # ========== 第一步：加载预训练权重 ==========
    # 使用权重加载器从指定路径加载权重
    # 这可能是从本地文件、云存储或网络下载
    loaded_params = loader.load(params_shape)
    
    # ========== 第二步：验证权重形状和数据类型 ==========
    # 确保加载的权重与期望的形状和数据类型完全匹配
    # 这对于模型正确性至关重要
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # ========== 第三步：过滤未加载的参数 ==========
    # 从加载的参数中移除jax.ShapeDtypeStruct（占位符），确保只返回实际加载的参数
    # 这样可以避免在后续处理中出现问题
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() 
         if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    """
    初始化训练状态函数
    
    这个函数负责创建和初始化整个训练过程所需的状态，包括：
    1. 模型参数（从预训练权重或随机初始化）
    2. 优化器状态
    3. 训练步数
    4. EMA（指数移动平均）参数
    5. 分片策略（用于多GPU训练）
    
    主要功能：
    - 支持从预训练权重初始化（如Pi0基础模型）
    - 支持从检查点恢复训练
    - 自动处理多GPU分片和内存优化
    - 配置冻结参数和可训练参数
    
    Args:
        config: 训练配置对象，包含模型、优化器、学习率等所有训练参数
        init_rng: 初始化随机数生成器，用于模型参数的随机初始化
        mesh: JAX分片网格，定义多GPU训练的设备布局
        resume: 是否从检查点恢复训练，影响初始化策略

    Returns:
        训练状态和分片信息的元组：
        - TrainState: 包含所有训练状态的对象
        - Sharding: 参数分片策略，用于多GPU训练
    """
    # ========== 第一步：创建优化器 ==========
    # 根据配置创建优化器（如AdamW），包含学习率调度策略
    # weight_decay_mask=None 表示对所有参数应用权重衰减
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        """
        初始化训练状态的内部函数
        
        这个函数是实际执行初始化工作的核心，它：
        1. 创建模型并初始化参数
        2. 合并预训练权重（如果提供）
        3. 设置参数精度（冻结参数使用bfloat16）
        4. 创建优化器状态
        5. 初始化EMA参数
        
        Args:
            rng: 随机数生成器，用于模型参数的随机初始化
            partial_params: 部分预训练参数（如Pi0基础模型权重）
                           None表示完全随机初始化

        Returns:
            初始化的训练状态对象
        """
        # ========== 第二步：分割随机数生成器 ==========
        # 为模型初始化创建专用的随机数生成器
        # 这确保了模型初始化的随机性与训练过程的随机性独立
        rng, model_rng = jax.random.split(rng)
        
        # ========== 第三步：创建并初始化模型 ==========
        # 根据配置创建模型实例（如Pi0模型）
        # 这会随机初始化所有模型参数
        model = config.model.create(model_rng)

        # ========== 第四步：合并预训练权重（如果提供） ==========
        if partial_params is not None:
            # 将模型分解为图定义和状态两部分
            # graphdef: 模型的结构定义（不可变）
            # state: 模型的参数状态（可变）
            graphdef, state = nnx.split(model)
            
            # 用预训练参数替换随机初始化的参数
            # 注意：如果partial_params不是state的子集，这里会报错
            state.replace_by_pure_dict(partial_params)
            
            # 重新合并图定义和更新后的状态
            model = nnx.merge(graphdef, state)

        # ========== 第五步：提取模型参数 ==========
        # 从模型中提取所有参数，用于训练和优化
        params = nnx.state(model)
        
        # ========== 第六步：设置参数精度 ==========
        # 将冻结的参数转换为bfloat16精度以节省内存
        # 可训练参数保持原有精度（通常是float32）
        params = nnx_utils.state_map(
            params, 
            config.freeze_filter,  # 过滤出需要冻结的参数
            lambda p: p.replace(p.value.astype(jnp.bfloat16))  # 转换为bfloat16
        )

        # ========== 第七步：创建训练状态对象 ==========
        return training_utils.TrainState(
            step=0,                                    # 初始训练步数为0
            params=params,                            # 模型参数（包含冻结和可训练参数）
            model_def=nnx.graphdef(model),            # 模型定义（用于保存检查点）
            tx=tx,                                    # 优化器实例
            opt_state=tx.init(params.filter(config.trainable_filter)),  # 优化器状态
            ema_decay=config.ema_decay,               # EMA衰减率（如0.999）
            ema_params=None if config.ema_decay is None else params,  # EMA参数初始值
        )

    # ========== 第八步：评估训练状态形状 ==========
    # 推断出init函数输出的形状和数据类型
    # 这是为了确定参数分片策略，避免实际分配内存
    train_state_shape = jax.eval_shape(init, init_rng)
    
    # ========== 第九步：创建FSDP分片策略 ==========
    # 根据训练状态形状和网格创建分片策略
    # FSDP（完全分片数据并行）将参数分布到多个GPU上
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    # ========== 第十步：处理恢复训练的情况 ==========
    if resume:
        # 如果是从检查点恢复，只需要返回形状和分片信息
        # 实际的参数会从检查点文件中加载
        return train_state_shape, state_sharding

    # ========== 第十一步：加载预训练权重 ==========
    # 从配置的权重加载器（如Pi0基础模型）加载预训练参数
    # 并验证这些参数与模型结构的兼容性
    partial_params = _load_weights_and_validate(
        config.weight_loader, 
        train_state_shape.params.to_pure_dict()
    )
    
    # ========== 第十二步：创建复制分片策略 ==========
    # 用于输入参数的分片策略（所有设备都复制相同的数据）
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # ========== 第十三步：实际初始化训练状态 ==========
    # 使用JAX JIT编译初始化函数，提高初始化效率
    train_state = jax.jit(
        init,                                           # 要编译的函数
        donate_argnums=(1,),                           # 捐赠partial_params缓冲区，节省内存
        in_shardings=replicated_sharding,              # 输入分片策略（复制到所有设备）
        out_shardings=state_sharding,                  # 输出分片策略（FSDP分片）
    )(init_rng, partial_params)

    # ========== 第十四步：返回完整的训练状态 ==========
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

    这是训练脚本的核心函数，负责协调整个训练流程。
    
    主要功能：
    1. 初始化日志记录和系统配置
    2. 验证训练配置的兼容性
    3. 创建分片网格和分片策略
    4. 初始化检查点管理器和WandB
    5. 创建数据加载器
    6. 初始化训练状态
    7. 执行训练循环
    8. 保存检查点

    Args:
        config: 训练配置对象，包含所有训练参数
    """
    # ========== 第一步：初始化日志记录 ==========
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    # ========== 第二步：验证批次大小与设备数量的兼容性 ==========
    # 确保批次大小能被设备数量整除，这样每个设备处理的批次大小相等
    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    # ========== 第三步：配置JAX编译缓存目录 ==========
    # 设置JAX编译缓存目录，加速重复编译
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    # ========== 第四步：初始化随机数生成器 ==========
    # 创建主随机数生成器，然后分割为训练和初始化用的随机数
    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    # ========== 第五步：创建分片网格和分片策略 ==========
    # 创建JAX分片网格，用于多GPU训练
    mesh = sharding.make_mesh(config.fsdp_devices)
    # 数据分片策略：按数据轴分片
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    # 复制分片策略：每个设备都有完整副本
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # ========== 第六步：初始化检查点管理器 ==========
    # 创建检查点管理器，负责保存和恢复训练状态
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,      # 检查点目录路径
        keep_period=config.keep_period,  # 检查点保留周期
        overwrite=config.overwrite,      # 是否覆盖现有检查点
        resume=config.resume,            # 是否恢复训练
    )

    # ========== 第七步：初始化WandB实验跟踪 ==========
    # 初始化Weights & Biases实验跟踪系统
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # ========== 第八步：创建数据加载器 ==========
    # 根据配置创建数据加载器，支持多种数据源和框架
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,  # 数据分片策略
        shuffle=True,            # 打乱数据顺序
    )
    data_iter = iter(data_loader)  # 创建数据迭代器
    batch = next(data_iter)        # 获取第一个批次用于验证
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # ========== 第九步：记录第一批次的图像用于检查 ==========
    # 将第一批次的图像记录到WandB，用于验证数据加载是否正确
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    # ========== 第十步：初始化训练状态 ==========
    # 创建训练状态，包括模型参数、优化器状态等
    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)  # 等待初始化完成
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    # ========== 第十一步：恢复训练状态（如果需要） ==========
    # 如果是从检查点恢复训练，从检查点恢复训练状态
    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    # ========== 第十二步：编译训练步骤函数 ==========
    # 使用JAX JIT编译训练步骤函数，提高训练效率
    ptrain_step = jax.jit(
        functools.partial(train_step, config),  # 部分应用配置参数
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),  # 输入分片策略
        out_shardings=(train_state_sharding, replicated_sharding),                # 输出分片策略
        donate_argnums=(1,),  # 捐赠训练状态缓冲区，节省内存
    )

    # ========== 第十三步：设置训练循环 ==========
    # 创建进度条，显示训练进度
    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    # ========== 第十四步：执行训练循环 ==========
    infos = []  # 存储训练指标
    for step in pbar:
        # 执行训练步骤：前向传播、反向传播、参数更新
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        
        # ========== 记录训练指标 ==========
        if step % config.log_interval == 0:
            # 计算平均指标并记录到日志和WandB
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        
        # ========== 获取下一个批次 ==========
        batch = next(data_iter)

        # ========== 保存检查点 ==========
        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    # ========== 第十五步：等待检查点管理器完成 ==========
    # 确保所有检查点都已保存完成
    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    # ========== 程序入口点 ==========
    # 使用配置命令行接口启动训练
    # cli() 会创建命令行接口，解析命令行参数，替换 TrainConfig 内容，然后返回 TrainConfig 对象给main 函数
    main(_config.cli())
