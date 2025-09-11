"""
Pi0/Pi0.5模型配置文件

这个文件定义了Pi0和Pi0.5模型的配置类，包括：
- 模型架构参数（动作维度、时间步长等）
- 语言模型变体选择（PaliGemma、动作专家网络）
- Pi0.5特有的配置选项（离散状态输入、adaRMSNorm等）
"""

import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    """
    Pi0/Pi0.5模型配置类
    
    这个配置类定义了Pi0和Pi0.5模型的所有超参数和架构选择。
    通过设置pi05=True可以启用Pi0.5的改进功能。
    """
    
    # ========== 基础模型参数 ==========
    dtype: str = "bfloat16"  # 数据类型，使用bfloat16可以节省内存并加速训练
    paligemma_variant: _gemma.Variant = "gemma_2b"  # PaliGemma语言模型变体，2B参数版本
    action_expert_variant: _gemma.Variant = "gemma_300m"  # 动作专家网络变体，300M参数版本

    # ========== 动作相关参数 ==========
    action_dim: int = 32  # 动作维度，通常对应机器人的关节数量
    action_horizon: int = 50  # 动作时间步长，即模型一次预测多少个时间步的动作
    max_token_len: int = None  # type: ignore  # 最大token长度，会在__post_init__中自动设置
    
    # ========== Pi0.5特有配置 ==========
    # Pi0.5相比Pi0有两个主要改进：
    # 1. 状态输入作为离散语言token的一部分，而不是连续输入作为后缀的一部分
    # 2. 动作专家网络使用adaRMSNorm来注入flow matching时间步
    pi05: bool = False  # 是否启用Pi0.5模式
    
    # 这个配置选项不直接被模型使用，但被ModelTransformFactory读取
    # 用于决定数据预处理的方式
    discrete_state_input: bool = None  # type: ignore  # 是否使用离散状态输入

    def __post_init__(self):
        """
        初始化后处理函数
        
        自动设置一些依赖其他参数的配置值：
        - max_token_len: Pi0.5使用200，Pi0使用48
        - discrete_state_input: 与pi05设置保持一致
        """
        if self.max_token_len is None:
            # Pi0.5需要更长的token序列来处理离散状态输入
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            # 离散状态输入与pi05模式保持一致
            object.__setattr__(self, "discrete_state_input", self.pi05)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        """
        返回模型类型
        
        Returns:
            ModelType.PI05 如果启用pi05模式，否则返回ModelType.PI0
        """
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        """
        创建Pi0模型实例
        
        Args:
            rng: 随机数生成器，用于模型参数初始化
            
        Returns:
            初始化好的Pi0模型实例
        """
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        """
        定义模型输入的数据规格
        
        这个方法定义了模型期望的输入数据格式，包括：
        - 图像数据：RGB图像，分辨率由IMAGE_RESOLUTION定义
        - 状态数据：机器人关节状态，维度为action_dim
        - 语言提示：token化的文本提示
        - 动作数据：动作序列，形状为[batch_size, action_horizon, action_dim]
        
        Args:
            batch_size: 批次大小，默认为1
            
        Returns:
            (observation_spec, action_spec): 观察和动作的数据规格
        """
        # 定义图像数据规格：RGB图像，3个通道
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        # 定义图像掩码规格：用于标识哪些图像是有效的
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            # 定义观察数据规格
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,        # 基础摄像头图像
                    "left_wrist_0_rgb": image_spec,  # 左手腕摄像头图像
                    "right_wrist_0_rgb": image_spec, # 右手腕摄像头图像
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,        # 基础摄像头掩码
                    "left_wrist_0_rgb": image_mask_spec,  # 左手腕摄像头掩码
                    "right_wrist_0_rgb": image_mask_spec, # 右手腕摄像头掩码
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),  # 机器人状态
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),  # token化提示
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),  # 提示掩码
            )
        # 定义动作数据规格：动作序列
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
