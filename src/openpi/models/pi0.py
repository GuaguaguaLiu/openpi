"""
Pi0/Pi0.5模型实现文件

这个文件包含了Pi0和Pi0.5模型的核心实现，包括：
- 模型架构定义（PaliGemma + 动作专家网络）
- 前缀和后缀嵌入方法
- 损失计算和动作采样
- Pi0.5特有的adaRMSNorm时间建模

Pi0.5相比Pi0的主要改进：
1. 使用离散状态输入而不是连续状态输入
2. 动作专家网络使用adaRMSNorm来注入flow matching时间步
3. 更长的token序列支持（200 vs 48）
"""

import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """
    创建注意力掩码
    
    这个函数从big_vision项目改编而来，用于创建不同类型的注意力掩码。
    Token可以关注到累积mask_ar小于或等于其自身的有效输入token。
    通过这种方式，mask_ar可以用于设置多种类型的注意力模式。
    
    示例：
    - [[1 1 1 1 1 1]]: 纯因果注意力
    - [[0 0 0 1 1 1]]: 前缀语言模型注意力。前3个token可以相互关注，
      后3个token具有因果注意力
    - [[1 0 1 0 1 0 0 1 0 0]]: 4个块之间的因果注意力。块内的token可以
      关注所有前面的块和同一块内的所有token
    
    Args:
        input_mask: bool[B, N] 输入掩码，True表示是输入的一部分，False表示填充
        mask_ar: bool[?B, N] 自回归掩码，True表示前面的token不能依赖它，
                False表示它与前一个token共享相同的注意力掩码
                
    Returns:
        bool[B, N, N] 注意力掩码，指定哪些token可以相互关注
    """
    # 形状对齐：mask_ar 的形状与 input_mask 对齐（B, N）
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    # 构造“块因果”索引：同一块共享相同的 cumsum 值，不同块严格因果
    cumsum = jnp.cumsum(mask_ar, axis=1)
    # 构造自注意力允许矩阵：query 位置的块索引 >= key 位置的块索引
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    # 仅在有效 token 上计算注意力（过滤 pad）
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """
    计算标量位置的正弦-余弦位置嵌入向量
    
    这个函数为标量位置（如时间步）生成位置嵌入，使用正弦和余弦函数的组合。
    这种位置编码方式在Transformer架构中广泛使用，能够很好地表示位置信息。
    
    Args:
        pos: 标量位置数组，形状为[b]
        embedding_dim: 嵌入维度，必须是偶数
        min_period: 最小周期
        max_period: 最大周期
        
    Returns:
        位置嵌入向量，形状为[b, embedding_dim]
        
    Raises:
        ValueError: 如果embedding_dim不是偶数
    """
    # 要求偶数维，便于拼接 sin/cos 两支
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    # 创建从0到1的分数，用于生成不同频率的正弦波
    # 频率按对数均匀分布（从 min_period 到 max_period）
    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    # 计算每个频率对应的周期
    period = min_period * (max_period / min_period) ** fraction
    # 计算正弦输入：位置 * 频率 * 2π
    # 计算 pos 与每个频率的乘积，作为 sin/cos 的输入
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    # 连接正弦和余弦值
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    """
    Pi0/Pi0.5模型类
    
    这是一个基于PaliGemma的视觉-语言-动作模型，能够根据图像观察和语言指令生成机器人动作。
    模型架构包括：
    1. PaliGemma语言模型：处理语言指令和图像特征
    2. 动作专家网络：专门处理动作生成
    3. 图像编码器：将图像转换为token
    4. 各种投影层：连接不同组件
    
    Pi0.5相比Pi0的改进：
    - 使用adaRMSNorm进行时间建模
    - 离散状态输入
    - 更长的token序列支持
    """
    # 设计要点（Pi0 vs Pi0.5）：
    # 1) 状态如何进入模型：
    #    - Pi0：连续状态作为后缀的一部分，经过线性层投影成一个 state token（见 embed_suffix 中 self.state_proj）。
    #    - Pi0.5：状态改为离散化为语言 token，直接拼进前缀序列（embed_prefix 阶段处理），因此不再额外添加 state token。
    # 2) 时间步如何注入动作专家：
    #    - Pi0：将时间步的位置编码（posemb_sincos）与动作 token 拼接后，经过 MLP 混合（action_time_mlp_*），不使用 adaRMS。
    #    - Pi0.5：不与动作拼接；使用 time MLP 生成条件向量 adarms_cond，通过 adaRMSNorm 调制动作专家（use_adarms）。
    # 3) 序列长度：
    #    - Pi0：较短（max_token_len=48）。
    #    - Pi0.5：更长（max_token_len=200），因为状态被离散化为 token 并拼入序列。
    # 4) 共同点：
    #    - 图像经 SigLIP 编码为图像 token；提示/状态（离散化后）经 LLM 的嵌入作为语言/状态 token。
    #    - 训练用 flow matching（compute_loss），推理用多步去噪（sample_actions）。

    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        """
        初始化Pi0模型
        
        Args:
            config: 模型配置
            rngs: 随机数生成器
        """
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05  # 是否启用Pi0.5模式
        
        # 获取PaliGemma和动作专家网络的配置
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        
        # 创建PaliGemma语言模型（含动作专家）
        # 说明：此处通过 nnx_bridge.ToNNX 将非 NNX 实现桥接进来。
        # PI0.5 差异：当 config.pi05=True 时，动作专家分支启用 adaRMS（use_adarms），用于时间调制。
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],  # 两个配置：语言模型和动作专家
                embed_dtype=config.dtype,
                adarms=config.pi05,  # Pi0.5使用adaRMSNorm
            )
        )
        # 初始化语言模型
        # use_adarms=[False, True] 表示只对动作专家分支使用 adaRMS（语言分支不使用）。
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        
        # 创建图像编码器（SigLIP）
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,  # 输出维度与PaliGemma匹配
                variant="So400m/14",  # SigLIP变体
                pool_type="none",  # 不使用池化
                scan=True,  # 使用扫描模式
                dtype_mm=config.dtype,
            )
        )
        # 初始化图像编码器
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        
        # 组合PaliGemma组件
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        
        # 动作输入投影层：将动作维度投影到专家网络维度
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        
        # Pi0.5 和 Pi0 使用不同的“时间步注入动作专家”的方式
        if config.pi05:
            # Pi0.5：时间 MLP 仅生成条件向量，供动作专家通过 adaRMSNorm 调制（不与动作直接拼接）。
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            # Pi0：连续状态作为单个 state token（通过 state_proj），
            #      时间位置编码与动作 token 在通道维拼接，经 MLP 混合（action_time_mlp_*）。
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            
        # 动作输出投影层：将专家网络输出投影回动作维度
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # 这个属性会被model.train()和model.eval()自动设置
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        # 前缀部分：图像 token + 语言/状态 token（若 pi05=True，离散状态已编码进 tokenized_prompt）。
        # 说明：
        # - 图像经 SigLIP -> token（互相可全连接关注），参与前缀的非因果注意力。
        # - 若存在 tokenized_prompt（语言/离散状态），加入到前缀，前缀之间是“全注意力”。
        # - Pi0 与 Pi0.5 在此的差异：Pi0.5 将“离散化后的状态”作为 token 融入这里；Pi0 不在此处放状态。
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # 图像 token 彼此之间为“全注意力”（非因果），因此 ar_mask 置 False
            ar_mask += [False] * image_tokens.shape[1]

        # 添加语言/离散状态（若已被离散化为 token）
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # 图像与语言/状态 token 之间采用全注意力（非因果）
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        # 后缀部分：状态（若是 Pi0 才添加 state token） + 动作 token + 时间注入（Pi0 vs Pi0.5 不同路径）。
        # 说明：
        # - Pi0：添加 state token（连续状态），再将动作 token 与时间位置编码拼接，经 MLP 混合。
        # - Pi0.5：不再添加 state token；时间经过 MLP 仅生成条件向量 adarms_cond，供专家网络的 adaRMSNorm 使用。
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            # Pi0：连续状态 -> 线性投影 -> 单一 state token，置于后缀最前，用于向动作专家提供全局状态。
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            # 对于后缀（state + actions），设置自回归块边界：
            #   第一个位置（state）为新块起点（True），后续动作序列另起一个块（下方追加）。
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        # 生成时间步的正余弦位置编码，用于时间条件。
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            # Pi0.5：时间仅用于生成条件向量，不与动作拼接，供 adaRMSNorm 使用。
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            # Pi0：将时间与动作在通道维拼接，再经 MLP 混合，直接作为动作专家输入。
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        # 设置后缀自回归块：第一个动作为新块起点（True），其后的动作沿用同一块（False）。
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        # 训练：flow matching 目标
        # 定义：
        #   x_t = t * noise + (1 - t) * actions
        #   u_t = noise - actions
        # 模型输出 v_t 逼近 u_t，使用 MSE(v_t, u_t)
        # 差异点：Pi0 与 Pi0.5 的前/后缀嵌入不同，但损失形式一致。
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        # 采样噪声与时间：t ~ Beta(1.5, 1) ∈ (0.001, 0.999)
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        # 构造插值目标 x_t 与监督目标 u_t
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # 前后缀一次前向：
        # - 前缀含图像 +（Pi0.5 时含离散状态/语言 token）
        # - 后缀含（Pi0: state token）+ 动作 token + 时间注入（Pi0.5 用 adaRMS 条件）
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        # 推理：多步去噪（Euler-like 形式）
        # 过程：
        #   1) 前缀预填充，构建 KV cache（图像 + 语言/离散状态）。
        #   2) while_loop 从 t=1 逐步向 t=0 演化：x_{t+dt} = x_t + dt * v_t
        #   3) 差异点：每步计算后缀时，Pi0 与 Pi0.5 在 embed_suffix 的时间注入方式不同。
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        # 构建前缀 token 与注意力 mask，并计算 KV cache 以加速后续自回归后缀计算。
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
