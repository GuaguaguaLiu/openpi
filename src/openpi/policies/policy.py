from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

# 类型别名：基础策略接口
BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    """OpenPi 策略类，支持 JAX 和 PyTorch 模型。

    该类是策略的核心实现，负责：
    1) 管理模型实例（JAX 或 PyTorch）；
    2) 应用输入和输出数据变换；
    3) 执行推理并返回动作；
    4) 处理不同框架的数据格式转换；
    5) 记录推理时间统计。

    支持两种模型类型：
    - JAX 模型：使用 JIT 编译优化，需要随机数生成器
    - PyTorch 模型：支持 GPU 加速，自动设备管理
    """

    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """初始化策略实例。

        Args:
            model: 用于动作采样的模型实例。
            rng: JAX 模型的随机数生成器密钥。PyTorch 模型忽略此参数。
            transforms: 推理前应用的输入数据变换序列。
            output_transforms: 推理后应用的输出数据变换序列。
            sample_kwargs: 传递给 model.sample_actions 的额外关键字参数。
            metadata: 与策略一起存储的额外元数据。
            pytorch_device: PyTorch 模型使用的设备（如 "cpu", "cuda:0"）。
                           仅在 is_pytorch=True 时相关。
            is_pytorch: 模型是否为 PyTorch 模型。如果为 False，假设为 JAX 模型。
        """
        self._model = model
        # 组合输入变换序列为单个变换函数
        self._input_transform = _transforms.compose(transforms)
        # 组合输出变换序列为单个变换函数
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            # PyTorch 模型设置
            self._model = self._model.to(pytorch_device)  # 移动到指定设备
            self._model.eval()  # 设置为评估模式
            self._sample_actions = model.sample_actions  # 直接使用模型方法
        else:
            # JAX 模型设置
            # 使用 JIT 编译优化采样方法
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            # 初始化随机数生成器
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        """执行策略推理，从观测生成动作。

        Args:
            obs: 输入观测字典，包含状态、图像等信息。
            noise: 可选的噪声数组，用于动作采样。

        Returns:
            包含动作和时间统计的字典。
        """
        # 创建输入副本，因为变换可能会就地修改输入
        inputs = jax.tree.map(lambda x: x, obs)
        # 应用输入变换
        inputs = self._input_transform(inputs)
        
        if not self._is_pytorch_model:
            # JAX 模型：创建批次并转换为 jax.Array
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            # 分割随机数生成器
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # PyTorch 模型：转换输入为 PyTorch 张量并移动到正确设备
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # 准备 sample_actions 的关键字参数
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            # 处理噪声参数
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # 如果噪声是 (action_horizon, action_dim)，添加批次维度
                noise = noise[None, ...]  # 使其变为 (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        # 从输入字典创建观测对象
        observation = _model.Observation.from_dict(inputs)
        
        # 记录推理开始时间
        start_time = time.monotonic()
        
        # 执行模型推理
        outputs = {
            "state": inputs["state"],  # 保持状态信息
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        
        # 计算模型推理时间
        model_time = time.monotonic() - start_time
        
        # 转换输出格式
        if self._is_pytorch_model:
            # PyTorch 模型：转换为 numpy 数组并移除批次维度
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            # JAX 模型：转换为 numpy 数组并移除批次维度
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        # 应用输出变换
        outputs = self._output_transform(outputs)
        
        # 添加策略时间统计
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,  # 推理时间（毫秒）
        }
        
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        """返回策略的元数据。"""
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """策略行为记录器，将策略的输入输出保存到磁盘。

    该类包装一个策略实例，在每次推理时记录输入和输出数据，
    用于分析策略行为、调试或数据收集。
    """

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        """初始化策略记录器。

        Args:
            policy: 要记录的策略实例。
            record_dir: 记录文件保存的目录路径。
        """
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        # 创建记录目录（如果不存在）
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0  # 记录步骤计数器

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        """执行推理并记录结果。

        Args:
            obs: 输入观测字典。

        Returns:
            策略推理结果。
        """
        # 执行策略推理
        results = self._policy.infer(obs)

        # 准备记录数据
        data = {"inputs": obs, "outputs": results}
        # 展平嵌套字典，使用 "/" 作为分隔符
        data = flax.traverse_util.flatten_dict(data, sep="/")

        # 生成输出文件路径
        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        # 保存数据到磁盘
        np.save(output_path, np.asarray(data))
        
        return results
