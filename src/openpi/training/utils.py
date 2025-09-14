from collections.abc import Callable
from typing import Any

from flax import nnx
from flax import struct
import jax
import optax

from openpi.models import model as _model
from openpi.shared import array_typing as at


@at.typecheck       # 启用类型检查（确保输入数据符合类型标注）
@struct.dataclass   # 让这个类变成不可变的JAX PyTree（关键！）
class TrainState:
    # 当前训练步数（从0开始）
    # at.Int[at.ArrayLike, ""] 表示：必须是整数类型的JAX数组
    step: at.Int[at.ArrayLike, ""]
    
    # 模型参数（所有权重和偏置）
    # nnx.State 是NNX的参数容器，本质是一个嵌套字典：
    # {
    #   'layer1': {'weight': Array(...), 'bias': Array(...)},
    #   'layer2': {...},
    #   ...
    # }
    params: nnx.State
    
    # 模型结构定义（不包含参数）
    # nnx.GraphDef 保存模型的计算图结构（类似模型的"骨架"）
    # _model.BaseModel 表示这个模型继承自某个基类
    model_def: nnx.GraphDef[_model.BaseModel]
    
    # 优化器状态（如Adam的动量、方差等）
    # optax.OptState 可能是一个嵌套结构，例如：
    # {
    #   'mu': {...},  # 动量
    #   'nu': {...}   # 二阶矩估计
    # }
    opt_state: optax.OptState
    
    # 优化器实例（如Adam）
    # struct.field(pytree_node=False) 表示这个字段不会被JAX视为PyTree的一部分
    # （即优化器定义不会被自动分片或传递）
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    
    # EMA（指数移动平均）的衰减率
    # None 表示不使用EMA
    ema_decay: float | None = struct.field(pytree_node=False)
    
    # EMA参数（平滑后的参数副本）
    # 初始为None，训练过程中会逐步更新
    # 计算方式：new_ema = decay * old_ema + (1-decay) * new_params
    ema_params: nnx.State | None = None


@at.typecheck
def tree_to_info(tree: at.PyTree, interp_func: Callable[[Any], str] = str) -> str:
    """Converts a PyTree into a human-readable string for logging. Optionally, `interp_func` can be provided to convert
    the leaf values to more meaningful strings.
    """
    tree, _ = jax.tree_util.tree_flatten_with_path(tree)
    return "\n".join(f"{jax.tree_util.keystr(path)}: {interp_func(value)}" for path, value in tree)


@at.typecheck
def array_tree_to_info(tree: at.PyTree) -> str:
    """Converts a PyTree of arrays into a human-readable string for logging."""
    return tree_to_info(tree, lambda x: f"{x.shape}@{x.dtype}")
