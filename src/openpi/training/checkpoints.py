"""
检查点管理模块 (Checkpoint Management Module)

这个模块负责机器学习训练过程中的检查点保存和恢复功能。
主要功能包括：
1. 初始化检查点目录
2. 保存训练状态（模型参数、优化器状态等）
3. 恢复训练状态
4. 管理数据标准化统计信息
5. 异步保存回调处理

检查点机制允许训练过程中断后能够从上次保存的状态继续训练，
这对于长时间训练任务非常重要。
"""

from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import logging
from typing import Protocol

from etils import epath
import jax
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    """
    初始化检查点目录和检查点管理器
    
    这个函数负责：
    1. 处理检查点目录的创建和清理
    2. 根据用户参数决定是覆盖、恢复还是报错
    3. 创建检查点管理器，配置不同的处理器
    4. 处理特殊情况（目录存在但没有有效检查点）
    
    Args:
        checkpoint_dir: 检查点保存目录路径
        keep_period: 检查点保留周期（每隔多少步保存一次）
        overwrite: 是否覆盖现有目录
        resume: 是否从现有检查点恢复训练
        
    Returns:
        tuple: (检查点管理器, 是否正在恢复训练)
    """
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    
    # 处理检查点目录已存在的情况
    if checkpoint_dir.exists():
        if overwrite:
            # 覆盖模式：删除现有目录并重新创建
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            # 恢复模式：标记为恢复训练
            resuming = True
        else:
            # 既不覆盖也不恢复：报错
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    # 确保目录存在
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 创建检查点管理器，配置不同的处理器
    # 检查点管理器就像一个"文件管理员"，负责保存和加载不同类型的文件
    mngr = ocp.CheckpointManager(
        checkpoint_dir,  # 检查点保存目录
        item_handlers={
            # item_handlers 定义了如何保存和加载不同类型的数据
            # 就像告诉管理员："这种文件用这种方式保存，那种文件用那种方式保存"
            
            "assets": CallbackHandler(),  
            # 用于保存资源文件（如标准化统计信息、配置文件等）
            # 这些文件通常比较小，不需要特殊处理
            
            "train_state": ocp.PyTreeCheckpointHandler(),  
            # 用于保存训练状态（包含模型参数、优化器状态、训练步数等）
            # PyTree是JAX的数据结构，可以包含嵌套的字典、列表、数组等
            # 这个处理器专门处理JAX的PyTree数据结构
            
            "params": ocp.PyTreeCheckpointHandler(),  
            # 用于保存模型参数（权重和偏置）
            # 虽然train_state也包含参数，但这里单独保存一份
            # 这样做的好处是可以单独加载模型参数，而不需要整个训练状态
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,  # 最多保留1个检查点
            # 这意味着只保留最新的检查点，删除旧的
            # 节省磁盘空间，但无法回退到之前的检查点
            
            keep_period=keep_period,  # 保存周期
            # 每隔多少步保存一次检查点
            # 比如keep_period=1000，表示每1000步保存一次
            
            create=False,  # 不自动创建
            # 如果检查点目录不存在，不会自动创建
            # 需要手动创建目录
            
            async_options=ocp.AsyncOptions(timeout_secs=7200),  # 异步保存超时时间（2小时）
            # 检查点保存是异步的，不会阻塞训练
            # 如果保存超过2小时还没完成，就超时
        ),
    )

    # 特殊情况处理：检查点目录存在但用户请求恢复训练，
    # 但训练运行没有保存到第一个检查点。在这种情况下，
    # 我们不希望训练脚本尝试恢复检查点，因为这会失败。
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
):
    """
    保存训练状态到检查点
    
    这个函数负责保存训练过程中的所有重要状态，包括：
    1. 训练状态（优化器状态、学习率等）
    2. 模型参数（用于推理）
    3. 数据标准化统计信息
    
    Args:
        checkpoint_manager: 检查点管理器
        state: 当前训练状态
        data_loader: 数据加载器（用于获取标准化统计信息）
        step: 当前训练步数
    """
    def save_assets(directory: epath.Path):
        """保存资源文件（主要是数据标准化统计信息）"""
        # 保存标准化统计信息
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)

    # 将可用于推理的参数分离到单独的项目中
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    
    # 构建要保存的项目字典
    items = {
        "assets": save_assets,  # 资源文件保存回调
        "train_state": train_state,  # 训练状态
        "params": {"params": params},  # 模型参数
    }
    checkpoint_manager.save(step, items)


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    """
    从检查点恢复训练状态
    
    这个函数负责从保存的检查点中恢复训练状态，包括：
    1. 训练状态（优化器状态、学习率等）
    2. 模型参数
    
    Args:
        checkpoint_manager: 检查点管理器
        state: 当前训练状态（用作模板）
        data_loader: 数据加载器（未使用，但保持接口一致性）
        step: 要恢复的步数，None表示恢复最新的检查点
        
    Returns:
        恢复的训练状态
    """
    del data_loader  # 数据加载器在此函数中未使用

    with at.disable_typechecking():
        # 将可用于推理的参数分离到单独的项目中
        train_state, params = _split_params(state)
        
        # 从检查点恢复状态
        restored = checkpoint_manager.restore(
            step,
            items={
                "train_state": train_state,
                "params": {"params": params},
            },
        )
    
    # 合并恢复的训练状态和参数
    return _merge_params(restored["train_state"], restored["params"])


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    """
    从检查点加载数据标准化统计信息
    
    这个函数用于在推理时加载训练时保存的数据标准化统计信息，
    确保推理时的数据预处理与训练时一致。
    
    Args:
        assets_dir: 资源文件目录路径
        asset_id: 资源ID（用于标识特定的标准化统计信息）
        
    Returns:
        标准化统计信息字典，如果不存在则返回None
    """
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    """回调函数协议，用于定义保存资源文件时的回调函数接口"""
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """
    异步回调处理器
    
    这是一个检查点处理器，用于异步调用任意函数。
    仅支持保存操作，不支持恢复操作。
    主要用于保存资源文件（如数据标准化统计信息）。
    """

    def save(self, directory: epath.Path, args: CallbackSave):
        """同步保存方法"""
        # 只在主进程中执行回调，避免多进程重复保存
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        """异步保存方法"""
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        """恢复方法（不支持）"""
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    """保存时的回调参数"""
    callback: Callback  # 要执行的回调函数


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): 
    """恢复时的回调参数（未实现）"""
    pass


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    """
    分离训练状态中的参数
    
    将训练状态中的参数分离出来，用于单独保存。
    优先使用EMA参数（如果存在），否则使用普通参数。
    
    Args:
        state: 完整的训练状态
        
    Returns:
        tuple: (不包含参数的训练状态, 分离出的参数)
    """
    if state.ema_params is not None:
        # 如果存在EMA参数，使用EMA参数
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        # 否则使用普通参数
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    """
    合并训练状态和参数
    
    将分离的参数重新合并到训练状态中。
    根据训练状态中是否已有参数来判断应该使用EMA参数还是普通参数。
    
    Args:
        train_state: 不包含参数的训练状态
        params: 要合并的参数字典
        
    Returns:
        完整的训练状态
    """
    # 恢复 `_split_params` 中的逻辑。
    # 假设 `params` 的存在意味着在分离时使用了EMA参数。
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])
