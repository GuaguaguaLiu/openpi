import logging
import os
import pathlib
from typing import Any

import jax.numpy as jnp

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """从训练好的检查点创建策略实例。

    该函数是策略创建的核心入口，负责：
    1) 自动检测模型类型（JAX vs PyTorch）；
    2) 加载模型权重和配置；
    3) 设置数据预处理和后处理管道；
    4) 配置设备（CPU/GPU）和推理参数。

    Args:
        train_config: 训练配置对象，包含模型架构、数据配置等信息。
        checkpoint_dir: 检查点目录路径，包含模型权重和元数据。
        repack_transforms: 可选的重新打包变换，将在其他变换之前应用。
        sample_kwargs: 传递给 `sample_actions` 方法的参数字典。如果未提供，将使用默认参数。
        default_prompt: 策略的默认提示词。如果输入数据中不存在提示词，将自动注入。
        norm_stats: 用于策略的归一化统计信息。如果未提供，将从检查点目录加载。
        pytorch_device: PyTorch 模型使用的设备（如 "cpu", "cuda", "cuda:0"）。
                       如果为 None 且 is_pytorch=True，将自动选择 "cuda"（如果可用）或 "cpu"。

    Returns:
        配置好的策略实例，可直接用于推理。

    Note:
        函数通过检查检查点目录中是否存在 "model.safetensors" 文件来自动检测模型类型：
        - 存在：PyTorch 模型
        - 不存在：JAX 模型
    """
    # 初始化重新打包变换（如果未提供则使用空组）
    repack_transforms = repack_transforms or transforms.Group()
    
    # 下载检查点（如果路径是远程 URL）
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # 通过检查 model.safetensors 文件的存在性来判断是否为 PyTorch 模型
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    logging.info("Loading model...")
    if is_pytorch:
        # PyTorch 模型加载路径
        model = train_config.model.load_pytorch(train_config, weight_path)
        # 将 PaliGemma 和专家网络的参数转换为 bfloat16 精度以节省内存
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        # JAX 模型加载路径：从 params 目录恢复参数
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    
    # 创建数据配置对象，包含数据变换和模型变换
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    
    # 加载归一化统计信息（如果未提供）
    if norm_stats is None:
        # 从检查点目录加载归一化统计信息，而不是从配置的资源目录，
        # 以确保策略使用与原始训练过程相同的归一化统计信息
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    # 为 PyTorch 模型确定设备
    if is_pytorch and pytorch_device is None:
        try:
            import torch
            # 自动选择 CUDA（如果可用）或 CPU
            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            # 如果 PyTorch 未安装，回退到 CPU
            pytorch_device = "cpu"

    # 创建并返回策略实例，配置完整的数据处理管道
    return _policy.Policy(
        model,  # 加载的模型实例
        transforms=[
            # 输入变换管道（按顺序应用）：
            *repack_transforms.inputs,  # 1. 重新打包变换
            transforms.InjectDefaultPrompt(default_prompt),  # 2. 注入默认提示词
            *data_config.data_transforms.inputs,  # 3. 数据变换（如图像预处理）
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),  # 4. 归一化
            *data_config.model_transforms.inputs,  # 5. 模型变换（如 tokenization）
        ],
        output_transforms=[
            # 输出变换管道（按顺序应用）：
            *data_config.model_transforms.outputs,  # 1. 模型输出变换（如 detokenization）
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),  # 2. 反归一化
            *data_config.data_transforms.outputs,  # 3. 数据输出变换
            *repack_transforms.outputs,  # 4. 重新打包输出变换
        ],
        sample_kwargs=sample_kwargs,  # 采样参数（如温度、top-k 等）
        metadata=train_config.policy_metadata,  # 策略元数据
        is_pytorch=is_pytorch,  # 模型类型标识
        pytorch_device=pytorch_device if is_pytorch else None,  # PyTorch 设备（仅 PyTorch 模型需要）
    )
