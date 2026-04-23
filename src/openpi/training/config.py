"""训练配置模块。

该模块定义了 OpenPi 项目的所有训练配置，包括：
- 基础配置类（AssetsConfig、DataConfig、TrainConfig）
- 数据配置工厂类（用于不同数据集）
- 预定义的训练配置（推理、微调、调试等）

参见 _CONFIGS 获取可用配置列表。
"""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.agi_policy as agi_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

# 类型别名：模型类型
ModelType: TypeAlias = _model.ModelType
# 类型别名：参数过滤器（解决 tyro 直接使用 nnx.filterlib.Filter 的问题）
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """资源文件配置类，确定用于设置数据管道的资源文件位置（如归一化统计信息）。

    这些资源文件将在检查点内的 `assets/asset_id` 目录下复制。

    可用于从不同的检查点（如基础模型检查点）或其他集中位置加载资源文件。
    例如，在微调期间从基础模型检查点加载 Trossen 机器人的归一化统计信息：

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```

    Attributes:
        assets_dir: 资源文件目录。如果未提供，将使用配置的 assets_dirs。
                   这对于从不同检查点（如基础模型检查点）或其他集中位置加载资源文件很有用。
        asset_id: 资源文件 ID。如果未提供，将使用 repo_id。
                 这允许用户引用描述不同机器人平台的资源文件。
    """

    # 资源文件目录。如果未提供，将使用配置的 assets_dirs
    # 这对于从不同检查点（如基础模型检查点）或其他集中位置加载资源文件很有用
    assets_dir: str | None = None

    # 资源文件 ID。如果未提供，将使用 repo_id
    # 这允许用户引用描述不同机器人平台的资源文件
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    """数据配置类，定义训练数据的加载和处理方式。

    该类包含数据集标识、变换管道、归一化设置等配置信息。

    Attributes:
        repo_id: LeRobot 仓库 ID。如果为 None，将创建假数据。
        asset_id: 包含数据资源的资源目录中的目录。
        norm_stats: 预计算的归一化统计信息。如果为 None，将不执行归一化。
        repack_transforms: 用于将输入从数据集特定格式转换为数据变换期望的通用格式。
        data_transforms: 数据变换，通常包括机器人特定的变换。将在数据归一化之前应用。
        model_transforms: 模型特定变换。将在数据归一化之后应用。
        use_quantile_norm: 如果为 True，将使用分位数归一化。否则使用标准 z-score 归一化。
        action_sequence_keys: 数据加载器用于生成动作序列的键名。序列长度由模型配置中的 `action_horizon` 字段定义。
        prompt_from_task: 如果为 True，将使用 LeRobot 数据集任务来定义提示。
        rlds_data_dir: 仅用于 RLDS 数据加载器（即目前仅用于 DROID）。
        action_space: DROID 数据集的动作空间。
        filter_dict_path: DROID 数据集的数据过滤器文件路径。
    """

    # LeRobot 仓库 ID。如果为 None，将创建假数据
    repo_id: str | None = None
    # 包含数据资源的资源目录中的目录
    asset_id: str | None = None
    # 包含预计算的归一化统计信息。如果为 None，将不执行归一化
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # 用于将输入从数据集特定格式转换为数据变换期望的通用格式
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # 数据变换，通常包括机器人特定的变换。将在数据归一化之前应用
    # 参见 `model.Observation` 和 `model.Actions` 了解归一化数据
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # 模型特定变换。将在数据归一化之后应用
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # 如果为 True，将使用分位数归一化。否则使用标准 z-score 归一化
    use_quantile_norm: bool = False

    # 数据加载器用于生成动作序列的键名。序列长度由模型配置中的 `action_horizon` 字段定义
    # 如果您的 LeRobot 数据集使用不同的键来表示动作，应调整此设置
    action_sequence_keys: Sequence[str] = ("actions",)

    # 如果为 True，将使用 LeRobot 数据集任务来定义提示
    prompt_from_task: bool = False

    # 仅用于 RLDS 数据加载器（即目前仅用于 DROID）
    rlds_data_dir: str | None = None
    # DROID 数据集的动作空间
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # RLDS 采样数据集列表（可配置每个数据集的权重与过滤规则）
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()


class GroupFactory(Protocol):
    """变换组工厂协议，用于创建数据变换组。"""
    
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """创建变换组。

        Args:
            model_config: 模型配置对象。

        Returns:
            数据变换组。
        """


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """标准 pi0 模型的模型变换工厂。

    该类为不同的模型类型（PI0、PI05、PI0_FAST）创建相应的模型变换组。
    变换包括图像调整、提示词注入、tokenization 等。

    Attributes:
        default_prompt: 如果提供，将确定模型使用的默认提示词。
    """

    # 如果提供，将确定模型使用的默认提示词
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """根据模型类型创建相应的变换组。

        Args:
            model_config: 模型配置对象。

        Returns:
            包含输入和输出变换的变换组。
        """
        match model_config.model_type:
            case _model.ModelType.PI0:
                # PI0 模型变换：注入提示词、调整图像、tokenize 提示词、填充状态和动作
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                # PI05 模型变换：与 PI0 类似，但支持离散状态输入
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                # PI0_FAST 模型变换：使用 FAST tokenizer，包含输入和输出变换
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    """数据配置工厂抽象基类。

    该类定义了创建数据配置的接口，子类需要实现具体的配置创建逻辑。
    主要用于为不同数据集创建相应的数据配置。

    Attributes:
        repo_id: LeRobot 仓库 ID。
        assets: 确定如何加载资源文件的配置。
        base_config: 将被工厂更新的基础配置。
    """

    # LeRobot 仓库 ID
    repo_id: str = tyro.MISSING
    # 确定如何加载资源文件的配置
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # 将被工厂更新的基础配置
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """创建数据配置。

        Args:
            assets_dirs: 资源文件目录路径。
            model_config: 模型配置对象。

        Returns:
            数据配置对象。
        """

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """创建基础数据配置。

        Args:
            assets_dirs: 资源文件目录路径。
            model_config: 模型配置对象。

        Returns:
            基础数据配置对象。
        """
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        """加载归一化统计信息。

        Args:
            assets_dir: 资源文件目录路径。
            asset_id: 资源文件 ID。

        Returns:
            归一化统计信息字典，如果加载失败则返回 None。
        """
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    """假数据配置类，用于调试和测试。
    
    该类创建一个简单的假数据配置，不加载真实数据集。
    """
    
    # 假数据仓库 ID
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """创建假数据配置。
        
        Args:
            assets_dirs: 资源文件目录路径（未使用）。
            model_config: 模型配置对象（未使用）。
            
        Returns:
            包含假数据仓库 ID 的数据配置对象。
        """
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    """简单数据配置类，用于创建基础的数据配置。
    
    该类允许用户自定义数据变换和模型变换工厂，提供灵活的数据配置方式。
    """
    
    # 数据变换工厂
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # 模型变换工厂
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """创建简单数据配置。
        
        Args:
            assets_dirs: 资源文件目录路径。
            model_config: 模型配置对象。
            
        Returns:
            包含自定义变换的数据配置对象。
        """
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    """LeRobot ALOHA 数据集配置类。
    
    该类专门用于配置 ALOHA 机器人的数据加载和变换。
    支持关节动作的增量转换、提示词注入、以及 ALOHA 空间到 Pi 内部运行时的转换。
    """
    
    # 如果为 True，将在传递给模型之前将关节维度转换为相对于当前状态的增量
    # 夹爪维度将保持绝对值
    use_delta_joint_actions: bool = True
    # 如果提供，将在输入数据中不存在 "prompt" 键时注入此提示词
    default_prompt: str | None = None
    # 如果为 True，这将把关节和夹爪值从标准 ALOHA 空间转换为
    # Pi 内部运行时使用的空间，该空间用于训练基础模型
    # 使用标准 ALOHA 数据的人应该将此设置为 True
    adapt_to_pi: bool = True

    # 重新打包变换
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # 用于从数据集中读取动作序列的动作键
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """创建 ALOHA 数据配置。
        
        Args:
            assets_dirs: 资源文件目录路径。
            model_config: 模型配置对象。
            
        Returns:
            配置好的 ALOHA 数据配置对象。
        """
        # 创建基础数据变换：ALOHA 输入和输出变换
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        
        # 如果启用增量关节动作，添加增量动作变换
        if self.use_delta_joint_actions:
            # 创建布尔掩码：前6个关节使用增量，夹爪保持绝对值
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # 创建模型变换（包含提示词注入等）
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        # 返回完整的数据配置
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """LeRobot LIBERO 数据集配置类。
    
    该类用于配置在数据管道各个部分应用的变换。
    对于您自己的数据集，您可以复制此类并根据下面的注释修改变换以匹配您的数据集。
    """

    # 是否应用额外的增量变换
    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """创建 LIBERO 数据配置。
        
        Args:
            assets_dirs: 资源文件目录路径。
            model_config: 模型配置对象。
            
        Returns:
            配置好的 LIBERO 数据配置对象。
        """
        # 重新打包变换仅应用于来自数据集的数据，而不在推理期间应用
        # 我们可以使用它使来自数据集的输入尽可能接近来自推理环境的输入（例如匹配键）
        # 下面，我们将数据集中的键（我们在数据转换脚本中定义的）匹配到
        # 我们在推理管道中使用的键（在 libero 的推理脚本中定义）
        # 对于您自己的数据集，首先确定您的环境传递给策略服务器的键，
        # 然后修改下面的映射，使您的数据集的键匹配到这些目标键
        # 重新打包变换在这里简单地重新映射键名
        # 注意：前面的是新键，后面的是旧键
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # 数据变换应用于来自数据集的数据和推理期间
        # 下面，我们定义进入模型的数据的变换（``inputs``）和
        # 从模型出来的数据的变换（``outputs``）（后者仅在推理期间使用）
        # 我们在 `libero_policy.py` 中定义了这些变换。您可以查看那里的详细注释，
        # 了解如何修改变换以匹配您的数据集。一旦您创建了自己的变换，
        # 您可以用自己的变换替换下面的变换
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # 一个额外的数据变换：pi0 模型在增量动作上训练（相对于每个动作块中的第一个状态）
        # 如果您的数据具有``绝对''动作（例如目标关节角度），
        # 您可以取消注释以下行以将动作转换为增量动作。唯一的例外是夹爪动作，它始终是绝对的
        # 在下面的示例中，我们将对前 6 个动作（关节）应用增量转换，
        # 并保持第 7 个动作（夹爪）不变，即绝对值
        # 在 Libero 中，数据集中的原始动作已经是增量动作，因此我们*不需要*
        # 应用单独的增量转换（这就是为什么它被注释掉）。根据您的数据集是否
        # 开箱即用地使用``绝对''或``增量''动作来选择是否应用此变换

        # LIBERO 已经将动作表示为增量，但我们有一些旧的 Pi0 检查点使用此额外的增量变换进行训练
        if self.extra_delta_transform:
            # 创建布尔掩码：前6个关节使用增量，夹爪保持绝对值
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # 模型变换包括诸如标记化提示词和动作目标等内容
        # 对于您自己的数据集，您不需要在这里更改任何内容
        model_transforms = ModelTransformFactory()(model_config)

        # 我们返回用于训练和推理的所有数据变换。这里不需要更改任何内容
        # replace是指替换掉 DataConfig 中的字段内容
        return dataclasses.replace(
            # 这里调的是 DataConfigFactory 的函数, 返回的是 DataConfig
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAgiDataConfig(DataConfigFactory):
    """LeRobot AGI 数据集配置类。
    
    该类专门用于配置 AGI 数据集的数据加载和变换。
    """

    # 是否应用额外的增量变换
    extra_delta_transform: bool = False
    only_use_head_camera: bool = True

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """创建 AGI 数据配置。
        
        Args:
            assets_dirs: 资源文件目录路径。
            model_config: 模型配置对象。
            
        Returns:
            配置好的 AGI 数据配置对象。
        """
        # 重新打包变换：将AGI数据集的键映射到标准键
        # 关键：将单数的'action'映射为复数的'actions'
        # 'observation.state',
        # 'action',
        # 'observation.velocity',
        # 'observation.images.cam_high',
        # 'observation.images.cam_low',
        # 'observation.images.cam_left_wrist',
        # 'observation.images.cam_right_wrist',
        # 'label',
        # 'timestamp',
        # 'frame_index',
        # 'episode_index',
        # 'index',
        # 'task_index',
        # 'action_is_pad',
        # 'task'
        # TODO 注意生成数据的时候生成的task 都是debug 有问题
        # TODO 需要注意 是把右侧映射成左侧, 并且映射完 只剩下左侧的内容(不会保留原来数据集的内容)
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {   
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # TODO 注意这个地方是自己定义的输入输出格式
        data_transforms = _transforms.Group(
            inputs=[agi_policy.AgiInputs(model_type=model_config.model_type, only_use_head_camera=self.only_use_head_camera)],
            outputs=[agi_policy.AgiOutputs()],
        )

        # 可选的增量变换
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # 模型变换
        model_transforms = ModelTransformFactory()(model_config)

        # 返回完整的数据配置
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """RLDS DROID 数据集配置类。
    
    用于在 DROID 上训练的配置，使用 RLDS 数据格式（用于在大型数据集上进行高效训练）。
    """

    # RLDS 数据目录路径
    rlds_data_dir: str | None = None
    # DROID 动作空间类型
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.

    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """创建 RLDS DROID 数据配置。
        
        Args:
            assets_dirs: 资源文件目录路径。
            model_config: 模型配置对象。
            
        Returns:
            配置好的 RLDS DROID 数据配置对象。
        """
        # 创建重新打包变换，将 RLDS 数据格式的键映射到标准键
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # 创建 DROID 数据变换
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        # 如果动作空间是关节位置，需要将绝对关节位置动作转换为增量动作进行训练
        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # 数据加载器返回绝对关节位置动作 -- 转换为增量动作进行训练
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # 创建模型变换
        model_transforms = ModelTransformFactory()(model_config)

        # 确保 RLDS 数据目录已设置
        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        # 返回完整的数据配置
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """LeRobot DROID 数据集配置类。
    
    用于自定义 DROID 数据集的示例数据配置，采用 LeRobot 格式。
    要将您的自定义 DROID 数据集（<10小时）转换为 LeRobot 格式，
    请参见 examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """创建 LeRobot DROID 数据配置。
        
        Args:
            assets_dirs: 资源文件目录路径。
            model_config: 模型配置对象。
            
        Returns:
            配置好的 LeRobot DROID 数据配置对象。
        """
        # 创建重新打包变换，将 LeRobot 格式的键映射到标准键
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        
        # 我们假设关节*速度*动作，因此我们*不应该*应用额外的增量变换
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        
        # 创建模型变换
        model_transforms = ModelTransformFactory()(model_config)

        # 返回完整的数据配置
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    """训练配置类，定义完整的训练流程参数。

    该类包含模型配置、数据配置、优化器设置、训练超参数等所有训练相关的配置信息。
    是训练脚本的主要配置入口。

    Attributes:
        name: 配置名称。必须唯一，用于引用此配置。
        project_name: 项目名称。
        exp_name: 实验名称。用于命名元数据和检查点目录。
        model: 模型配置。定义模型架构和参数。
        weight_loader: 权重加载器，用于从磁盘加载预训练权重。
        pytorch_weight_path: PyTorch 检查点的可选路径。
        pytorch_training_precision: PyTorch 训练精度。
        lr_schedule: 学习率调度器配置。
        optimizer: 优化器配置。
        ema_decay: 指数移动平均衰减率。
        freeze_filter: 指定哪些权重应该被冻结。
        data: 确定要训练的数据。
        assets_base_dir: 配置资源文件的基础目录。
        checkpoint_base_dir: 检查点的基础目录。
        seed: 训练期间随机生成器使用的随机种子。
        batch_size: 全局批次大小。
        num_workers: 数据加载器使用的工作进程数。
        num_train_steps: 要运行的训练步数（批次）。
        log_interval: 记录训练指标的频率（步数）。
        save_interval: 保存检查点的频率（步数）。
        keep_period: 如果设置，匹配 step % keep_period == 0 的现有检查点将不会被删除。
        overwrite: 如果为 True，将在检查点目录已存在时覆盖它。
        resume: 如果为 True，将从最后一个检查点恢复训练。
        wandb_enabled: 如果为 True，将启用 wandb 日志记录。
        policy_metadata: 用于传递给策略服务器的元数据。
        fsdp_devices: 如果值大于 1，将启用 FSDP 并在指定数量的设备上分片。
    """

    # 配置名称。必须唯一，用于引用此配置
    name: tyro.conf.Suppress[str]
    # 项目名称
    project_name: str = "openpi"
    # 实验名称。用于命名元数据和检查点目录
    exp_name: str = tyro.MISSING

    # 定义模型配置。某些属性（action_dim、action_horizon 和 max_token_len）由所有模型共享
    # 参见 BaseModelConfig。特定模型实现（如 Pi0Config）继承自 BaseModelConfig 并可能定义其他属性
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # 权重加载器可以在模型初始化后选择性地从磁盘加载（可能是部分的）权重
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # 用于加载权重的 PyTorch 检查点的可选路径
    pytorch_weight_path: str | None = None

    # PyTorch 训练精度
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    # 学习率调度器配置
    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    # 优化器配置
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    # 指数移动平均衰减率
    ema_decay: float | None = 0.99

    # 指定哪些权重应该被冻结
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # 确定要训练的数据
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # 配置资源文件的基础目录（如归一化统计信息）
    assets_base_dir: str = "./assets"
    # 检查点的基础目录
    checkpoint_base_dir: str = "./checkpoints"

    # 训练期间随机生成器使用的随机种子
    seed: int = 42
    # 全局批次大小
    batch_size: int = 32
    # 数据加载器使用的工作进程数。增加此数字将加快数据加载速度，但会增加内存和 CPU 使用率
    num_workers: int = 2
    # 要运行的训练步数（批次）
    num_train_steps: int = 30_000

    # 记录训练指标的频率（步数）
    log_interval: int = 100
    # 保存检查点的频率（步数）
    save_interval: int = 1000
    # 如果设置，匹配 step % keep_period == 0 的现有检查点将不会被删除
    keep_period: int | None = 5000

    # 如果为 True，将在检查点目录已存在时覆盖它
    overwrite: bool = False
    # 如果为 True，将从最后一个检查点恢复训练
    resume: bool = False

    # 如果为 True，将启用 wandb 日志记录
    wandb_enabled: bool = True

    # 用于传递给策略服务器的元数据
    policy_metadata: dict[str, Any] | None = None

    # 如果值大于 1，将启用 FSDP 并在指定数量的设备上分片
    # 总体设备内存将减少，但训练可能会变慢
    # 例如，如果总设备数为 4，fsdp 设备数为 2，则模型将分片到 2 个设备并在 2 组设备之间运行数据并行
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """获取此配置的资源文件目录。

        Returns:
            资源文件目录的绝对路径。
        """
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """获取此配置的检查点目录。

        Returns:
            检查点目录的绝对路径。

        Raises:
            ValueError: 如果 exp_name 未设置。
        """
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """获取可训练参数的过滤器。

        Returns:
            可训练参数的过滤器。
        """
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        """初始化后验证配置参数。

        Raises:
            ValueError: 如果同时设置了 resume 和 overwrite。
        """
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# 预定义的训练配置列表
# 使用 `get_config` 函数通过名称获取配置
_CONFIGS = [
    #
    # 推理 ALOHA 配置
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # 推理 DROID 配置
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # 微调 LIBERO 配置
    #
    # 这些训练配置定义了在您自己的数据集上微调基础模型的超参数
    # 它们用于定义关键元素，如您正在训练的数据集、您正在使用的基础检查点，
    # 以及其他超参数，如运行多少训练步数或使用什么学习率
    # 对于您自己的数据集，您可以复制此类并根据下面的注释修改数据集名称和数据变换
    TrainConfig(
        # 更改名称以反映您的模型和数据集
        name="pi0_libero",
        # 这里您定义模型配置 -- 在此示例中，我们使用 pi0 作为模型
        # 架构并执行*完整*微调。在下面的示例中，我们展示如何修改
        # 此配置以执行*低内存*（LORA）微调并使用 pi0-FAST 作为替代架构
        model=pi0_config.Pi0Config(),
        # 这里您定义您正在训练的数据集。在此示例中，我们使用 Libero
        # 数据集。对于您自己的数据集，您可以更改 repo_id 指向您的数据集
        # 还要修改 DataConfig 以使用您为数据集创建的新配置
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # 此标志确定我们是否从 LeRobot 数据集的 ``task`` 字段加载提示词（即任务指令）
                # 如果设置为 True，提示词将出现在输入字典中名为 ``prompt`` 的字段中
                # 推荐设置为 True
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # 这里您定义要加载哪个预训练检查点来初始化模型
        # 这应该与您在上面选择的模型配置匹配 -- 即在这种情况下，我们使用 pi0 基础模型
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # 下面您可以定义其他超参数，如学习率、训练步数等
        # 查看基础 TrainConfig 类以获取可用超参数的完整列表
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # 这是为 LoRA 微调加载 pi0 模型的示例
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # 冻结过滤器定义在训练期间应该冻结哪些参数
        # 我们在模型配置中有一个便利函数，它返回给定模型配置的默认冻结过滤器
        # 用于 LoRA 微调。只需确保它与您在上面选择的模型配置匹配
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # 为 LoRA 微调关闭 EMA
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # 这是为完整微调加载 pi0-FAST 模型的示例
        # 修改 action_dim 和 action_horizon 以匹配您的数据集（动作范围等于所需的动作块长度）
        # max_token_len 是模型可以处理的最大（非图像）token 数量
        # 这包括标记化的提示词、本体感受状态和（FAST 标记化的）动作 token
        # 选择此值太小可能会在序列末尾截断 token（代码将抛出警告），
        # 而选择太大将浪费内存（因为我们将每个批次元素填充到 max_token_len）
        # 一个好的经验法则是单臂机器人使用约 180，双臂机器人使用约 250
        # 一般来说，首先在较低的一侧犯错，如果您在训练期间看到许多警告被抛出，则可能增加该值
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # 注意，我们在这里加载 pi0-FAST 基础模型检查点
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # 这是为 LoRA 微调加载 pi0-FAST 模型的示例
        # 有关设置 action_dim、action_horizon 和 max_token_len 的信息，请参见上面的注释
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # 再次确保在提取指定在 LoRA 微调期间应该冻结哪些参数的冻结过滤器时
        # 与上面的模型配置匹配
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # 为 LoRA 微调关闭 EMA
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    #
    # 微调 ALOHA 配置
    #
    # 这是一个测试配置，用于说明如何在自定义 LeRobot 数据集上训练
    # 有关如何转换和训练您自己的 ALOHA 数据集的说明，请参见 examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # 微调 DROID 配置
    #
    TrainConfig(
        # 此配置用于在*完整* DROID 数据集上微调 pi0-FAST-base
        # 我们使用 RLDS 数据加载来使在这个大型数据集上的训练变得可行
        # 有关在您自己的 DROID 数据集上微调的信息，请参见下面
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # 将此设置为您的 DROID RLDS 数据集的路径（`droid` 目录的父目录）
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k 步应该足够，在 8x H100s 上需要约 2 天
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # 重要：RLDS DataLoader 需要 num_workers=0，内部处理多进程
    ),
    TrainConfig(
        # 此配置用于在*完整* DROID 数据集上微调 pi05
        # 我们使用 RLDS 数据加载来使在这个大型数据集上的训练变得可行
        # 有关在您自己的 DROID 数据集上微调的信息，请参见下面
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # 将此设置为您的 DROID RLDS 数据集的路径（`droid` 目录的父目录）
            rlds_data_dir="/mnt/pi-data/kevin",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # 重要：RLDS DataLoader 需要 num_workers=0，内部处理多进程
    ),
    TrainConfig(
        # 此配置用于在自定义（较小）DROID 数据集上微调 pi05-DROID
        # 这里，我们使用 LeRobot 数据格式（如所有其他微调示例）
        # 要将您的自定义 DROID 数据集（<10小时）转换为 LeRobot 格式，
        # 请参见 examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 使用 32 维动作进行训练
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # 替换为您的自定义 DROID LeRobot 数据集仓库 ID
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # 重要：在微调期间重用原始 DROID 归一化统计信息！
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    #
    # ALOHA 仿真配置。此配置用于演示如何在简单仿真环境中训练。
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # 调试配置
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    TrainConfig(
        name="pi05_libero_debug",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/home/tione/notebook/workspace/rickyyzliu/code/openpi/checkpoints_pytorch",
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="agi_debug",  # 名字要注意 根据名字加载配置
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotAgiDataConfig(  # 这个是自己单独定义的
            repo_id="jaka_joint_sim",  # 注意: 从HF_LEROBOT_HOME/test_agi_rickyyzliu_0915 下面去加载数据
            # repo_id="test_agi_210",  # 注意: 从HF_LEROBOT_HOME/test_agi_rickyyzliu_0915 下面去加载数据
            base_config=DataConfig(prompt_from_task=True, action_sequence_keys=("action",)),  # 注意: action_sequence_keys=("action",) 这里的action要和你数据集里action名字保持一致，小心actions！
            extra_delta_transform=False,
            only_use_head_camera=True,
        ),
        batch_size=128,
        num_workers=128,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/home/tione/notebook/workspace/rickyyzliu/code/openpi/checkpoints_pytorch",
        num_train_steps=30_000,
    ),
    # #
    # # 微调 LIBERO 配置
    # #
    # # 这些训练配置定义了在您自己的数据集上微调基础模型的超参数
    # # 它们用于定义关键元素，如您正在训练的数据集、您正在使用的基础检查点，
    # # 以及其他超参数，如运行多少训练步数或使用什么学习率
    # # 对于您自己的数据集，您可以复制此类并根据下面的注释修改数据集名称和数据变换
    # TrainConfig(
    #     # 更改名称以反映您的模型和数据集
    #     name="pi0_libero",
    #     # 这里您定义模型配置 -- 在此示例中，我们使用 pi0 作为模型
    #     # 架构并执行*完整*微调。在下面的示例中，我们展示如何修改
    #     # 此配置以执行*低内存*（LORA）微调并使用 pi0-FAST 作为替代架构
    #     model=pi0_config.Pi0Config(),
    #     # 这里您定义您正在训练的数据集。在此示例中，我们使用 Libero
    #     # 数据集。对于您自己的数据集，您可以更改 repo_id 指向您的数据集
    #     # 还要修改 DataConfig 以使用您为数据集创建的新配置
    #     data=LeRobotLiberoDataConfig(
    #         repo_id="physical-intelligence/libero",
    #         base_config=DataConfig(
    #             # 此标志确定我们是否从 LeRobot 数据集的 ``task`` 字段加载提示词（即任务指令）
    #             # 如果设置为 True，提示词将出现在输入字典中名为 ``prompt`` 的字段中
    #             # 推荐设置为 True
    #             prompt_from_task=True,
    #         ),
    #         extra_delta_transform=True,
    #     ),
    #     # 这里您定义要加载哪个预训练检查点来初始化模型
    #     # 这应该与您在上面选择的模型配置匹配 -- 即在这种情况下，我们使用 pi0 基础模型
    #     weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
    #     # 下面您可以定义其他超参数，如学习率、训练步数等
    #     # 查看基础 TrainConfig 类以获取可用超参数的完整列表
    #     num_train_steps=30_000,
    # ),
    #
    # RoboArena 配置
    #
    *roboarena_config.get_roboarena_configs(),
    *polaris_config.get_polaris_configs(),
]

# 验证配置名称的唯一性
if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")

# 创建配置名称到配置对象的映射字典
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    """创建命令行接口，允许用户选择配置。

    Returns:
        用户选择的训练配置。
    """
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """通过名称获取配置。

    Args:
        config_name: 配置名称。

    Returns:
        对应的训练配置对象。

    Raises:
        ValueError: 如果配置名称不存在，会提供最接近的匹配建议。
    """
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
