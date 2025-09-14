"""
数据加载器模块 (Data Loader Module)

这个模块负责机器学习训练过程中的数据加载和预处理功能。
主要功能包括：
1. 定义数据集和数据加载器的接口协议
2. 支持多种数据源（LeRobot数据集、RLDS数据集、假数据）
3. 数据变换和标准化处理
4. 支持PyTorch和JAX两种框架
5. 分布式训练支持
6. 多进程数据加载

数据加载器是训练流程中的关键组件，负责高效地从各种数据源加载数据，
并进行必要的预处理和批处理。
"""

# 导入必要的库和模块
from collections.abc import Iterator, Sequence  # 迭代器和序列类型
import logging                                  # 日志记录
import multiprocessing                         # 多进程支持
import os                                      # 操作系统接口
import typing                                  # 类型提示
from typing import Literal, Protocol, SupportsIndex, TypeVar  # 类型相关

import jax                                     # JAX深度学习框架
import jax.numpy as jnp                       # JAX的NumPy接口
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset  # LeRobot数据集
import numpy as np                            # NumPy数值计算库
import torch                                  # PyTorch深度学习框架

import openpi.models.model as _model          # OpenPI模型定义
import openpi.training.config as _config      # 训练配置
from openpi.training.droid_rlds_dataset import DroidRldsDataset  # DROID RLDS数据集
import openpi.transforms as _transforms       # 数据变换

# 类型变量，用于泛型类型定义
T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """
    数据集接口协议
    
    定义了支持随机访问的数据集接口，类似于PyTorch的Dataset。
    子类需要实现索引访问和长度查询功能。
    """

    def __getitem__(self, index: SupportsIndex) -> T_co:
        """通过索引获取数据项"""
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        """返回数据集大小"""
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """
    可迭代数据集接口协议
    
    定义了支持迭代访问的数据集接口，适用于流式数据或大型数据集。
    子类需要实现迭代器和长度查询功能。
    """

    def __iter__(self) -> Iterator[T_co]:
        """返回数据集的迭代器"""
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        """返回数据集大小"""
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """
    数据加载器接口协议
    
    定义了数据加载器的标准接口，负责从数据集中批量加载数据。
    子类需要实现数据配置获取和迭代器功能。
    """

    def data_config(self) -> _config.DataConfig:
        """获取数据加载器的配置信息"""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        """返回数据加载器的迭代器，用于批量获取数据"""
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    """
    变换数据集包装器
    
    对原始数据集应用一系列数据变换，如标准化、数据增强等。
    支持链式变换，可以组合多个变换函数。
    """
    
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        """
        初始化变换数据集
        
        Args:
            dataset: 原始数据集
            transforms: 要应用的数据变换序列
        """
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)  # 组合多个变换函数

    def __getitem__(self, index: SupportsIndex) -> T_co:
        """获取变换后的数据项"""
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    """
    可迭代变换数据集包装器
    
    对可迭代数据集应用数据变换，支持批处理和非批处理模式。
    在批处理模式下，会将批次拆分为单个样本进行变换，然后重新组合。
    """
    
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        """
        初始化可迭代变换数据集
        
        Args:
            dataset: 原始可迭代数据集
            transforms: 要应用的数据变换序列
            is_batched: 是否为批处理模式
        """
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        """迭代数据集，应用变换"""
        for sample in self._dataset:
            if self._is_batched:
                # 变换函数设计为应用于单个样本，所以需要将批次拆分为单个样本
                # 然后对每个样本单独应用变换
                batch_size = next(v.shape[0] for v in sample.values())

                # 使用tree_map将批次拆分为单个样本
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # 对每个样本应用变换
                transformed = [self._transform(s) for s in individual_samples]

                # 使用tree_map重新组合批次
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self._dataset)


class FakeDataset(Dataset):
    """
    假数据集
    
    用于测试和调试的数据集，根据模型配置生成随机数据。
    生成的数据符合模型期望的输入格式和数据类型。
    """
    
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        """
        初始化假数据集
        
        Args:
            model_config: 模型配置，用于确定数据格式
            num_samples: 生成的样本数量
        """
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        """
        生成指定索引的假数据
        
        Args:
            index: 数据项的索引
            
        Returns:
            包含观察数据和动作数据的字典
        """
        # 使用索引作为随机种子，确保相同索引生成相同数据
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            """
            根据数据规格生成随机数据
            
            Args:
                spec: 数据规格，包含形状和数据类型
                
            Returns:
                符合规格的随机数据
            """
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # 移除批次维度（第一个维度）
            shape = spec.shape[1:]
            
            # 根据数据类型生成不同的随机数据
            if spec.dtype == jnp.float32:
                # 浮点数：生成[-1, 1]范围内的均匀分布随机数
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                # 整数：生成[0, 2048)范围内的随机整数（用于token）
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            # 其他类型：生成零填充数据
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        # 生成观察数据和动作数据
        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),  # 展开观察数据字典
            "actions": action,        # 添加动作数据
        }

    def __len__(self) -> int:
        """返回数据集大小"""
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """
    创建PyTorch数据集
    
    这个函数是数据集创建的核心，负责：
    1. 根据配置选择合适的数据集类型
    2. 从HuggingFace Hub下载和缓存数据集
    3. 配置动作序列的时间戳
    4. 应用必要的预处理变换
    
    支持的数据集类型：
    - LeRobot数据集：从HuggingFace Hub下载的真实机器人数据
    - 假数据集：用于测试和调试的随机数据
    
    Args:
        data_config: 数据配置，包含数据集信息、变换设置等
        action_horizon: 动作序列长度，模型一次预测多少步动作
        model_config: 模型配置，决定数据格式和预处理方式
        
    Returns:
        创建的数据集，可以用于训练
        
    Raises:
        ValueError: 如果repo_id未设置
    """
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    
    # 检查是否为假数据集
    if repo_id == "fake":
        # 创建假数据集，用于测试和调试
        return FakeDataset(model_config, num_samples=1024)

    # 获取数据集元数据
    # 这包含了数据集的详细信息，如帧率、任务列表等
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    
    # 自动从 HuggingFace Hub 下载和缓存数据集
    # LeRobotDataset会自动处理下载、缓存、数据格式转换等
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        # 配置动作序列的时间戳
        # 将动作序列的索引转换为时间戳（秒）
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] 
            for key in data_config.action_sequence_keys
        },
    )

    # 如果需要从任务生成提示，应用相应的变换
    # 这会将任务描述转换成语言指令
    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    """
    创建RLDS数据集
    
    RLDS (Robotics Learning Dataset) 是Google开发的一种机器人学习数据集格式。
    这个函数专门用于创建RLDS格式的数据集，目前主要支持DROID数据集。
    
    RLDS数据集的特点：
    - 使用TensorFlow的TFRecord格式存储
    - 支持流式数据访问
    - 适合大规模机器人数据
    - 支持数据过滤和预处理
    
    Args:
        data_config: 数据配置，包含RLDS数据集路径和设置
        action_horizon: 动作序列长度，模型一次预测多少步动作
        batch_size: 批次大小，用于批处理数据
        shuffle: 是否打乱数据顺序，提高训练效果
        
    Returns:
        创建的RLDS数据集，可以用于训练
        
    Note:
        目前只支持DROID数据集格式的RLDS数据集
    """
    # 目前只支持DROID格式的RLDS数据集
    # DroidRldsDataset是专门为DROID数据集设计的RLDS数据集实现
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,        # RLDS数据目录路径
        batch_size=batch_size,                     # 批次大小
        shuffle=shuffle,                           # 是否打乱数据
        action_chunk_size=action_horizon,          # 动作序列长度
        action_space=data_config.action_space,     # 动作空间定义
        filter_dict_path=data_config.filter_dict_path,  # 数据过滤配置路径
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """
    对数据集应用数据变换
    
    这个函数是数据预处理的核心，负责将原始数据转换成模型可以使用的格式。
    按照配置的顺序应用各种数据变换，包括：
    1. 重新打包变换：键名映射、数据重组
    2. 数据变换：数据增强、格式转换
    3. 标准化：数据归一化，使用统计信息
    4. 模型变换：模型特定的预处理
    
    
    变换顺序很重要，因为每个变换都依赖于前一个变换的输出。
    
    Args:
        dataset: 原始数据集，需要应用变换
        data_config: 数据配置，包含所有变换设置
        skip_norm_stats: 是否跳过标准化统计信息（用于调试）
        
    Returns:
        变换后的数据集，可以直接用于训练
        
    Raises:
        ValueError: 如果标准化统计信息未找到且未跳过
    """
    # 1. 初始化标准化统计信息
    norm_stats = {}
    
    # 2. 检查是否需要标准化统计信息
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            # 标准化统计信息未找到，需要先计算
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    # 3. 创建变换后的数据集
    # 按照特定顺序应用变换，确保数据格式正确
    return TransformedDataset(
        dataset,
        [
            # 1. 重新打包变换：键名映射、数据重组
            *data_config.repack_transforms.inputs,
            
            # 2. 数据变换：数据增强、格式转换
            *data_config.data_transforms.inputs,
            
            # 3. 标准化：使用统计信息进行数据归一化
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            
            # 4. 模型变换：模型特定的预处理
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """
    对可迭代数据集应用数据变换
    
    类似于transform_dataset，但适用于可迭代数据集，支持批处理模式。
    
    Args:
        dataset: 原始可迭代数据集
        data_config: 数据配置
        skip_norm_stats: 是否跳过标准化统计信息
        is_batched: 是否为批处理模式
        
    Returns:
        变换后的可迭代数据集
    """
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,  # 重新打包变换
            *data_config.data_transforms.inputs,    # 数据变换
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),  # 标准化
            *data_config.model_transforms.inputs,   # 模型变换
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """
    创建数据加载器
    
    这是数据加载的主要入口函数，负责根据训练配置创建合适的数据加载器。
    支持两种数据源：
    1. PyTorch数据集：LeRobot数据集、假数据集等
    2. RLDS数据集：DROID数据集等
    
    函数会根据数据源类型自动选择合适的数据加载器实现。
    
    Args:
        config: 训练配置，包含模型、数据、训练参数等所有配置
        sharding: JAX数据分片配置，用于多GPU训练（仅JAX框架使用）
        shuffle: 是否打乱数据顺序，提高训练效果
        num_batches: 返回的批次数量限制，None表示使用整个数据集
        skip_norm_stats: 是否跳过数据标准化（用于调试）
        framework: 使用的框架（"jax" 或 "pytorch"）
        
    Returns:
        创建的数据加载器，可以迭代获取训练批次
        
    Note:
        这个函数是训练脚本中数据加载的主要入口点
    """
    # 第一步：创建数据配置
    # 从训练配置中提取数据配置，并应用资源和模型设置: 这里的 config.data 实际上是 LeRobotLiberoDataConfig
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    # 第二步：根据数据源类型选择相应的数据加载器
    if data_config.rlds_data_dir is not None:
        # 使用RLDS数据加载器（如DROID数据集）
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,  # 动作序列长度
            batch_size=config.batch_size,                # 批次大小
            sharding=sharding,                           # 分片配置
            shuffle=shuffle,                             # 是否打乱数据
            num_batches=num_batches,                     # 批次数量限制
            skip_norm_stats=skip_norm_stats,             # 是否跳过标准化
            framework=framework,                         # 使用的框架
        )
    
    # 使用PyTorch数据加载器（如LeRobot数据集、假数据集）
    return create_torch_data_loader(
        data_config,
        model_config=config.model,                       # 模型配置
        action_horizon=config.model.action_horizon,      # 动作序列长度
        batch_size=config.batch_size,                    # 批次大小
        sharding=sharding,                               # 分片配置
        shuffle=shuffle,                                 # 是否打乱数据
        num_batches=num_batches,                         # 批次数量限制
        num_workers=config.num_workers,                  # 数据加载工作进程数
        seed=config.seed,                                # 随机种子
        skip_norm_stats=skip_norm_stats,                 # 是否跳过标准化
        framework=framework,                             # 使用的框架
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """
    创建PyTorch数据加载器
    
    这个函数是数据加载的核心，负责：
    1. 创建数据集（从原始数据到可用的数据格式）
    2. 应用数据变换（标准化、键名映射等）
    3. 处理分布式训练（PyTorch DDP vs JAX）
    4. 创建实际的数据加载器
    
    支持两种框架：
    - PyTorch: 使用PyTorch的分布式训练
    - JAX: 使用JAX的分片机制
    
    Args:
        data_config: 数据配置，包含数据集信息、变换设置等
        model_config: 模型配置，决定数据格式和预处理方式
        action_horizon: 动作序列长度，模型一次预测多少步动作
        batch_size: 全局批次大小，会被分配到各个设备上
        sharding: JAX数据分片配置，用于多GPU训练
        skip_norm_stats: 是否跳过数据标准化（用于调试）
        shuffle: 是否打乱数据顺序
        num_batches: 返回的批次数量，None表示使用整个数据集
        num_workers: 数据加载工作进程数，0表示在主进程中执行
        seed: 随机种子，用于数据打乱
        framework: 使用的框架（"jax" 或 "pytorch"）
        
    Returns:
        创建的数据加载器，可以迭代获取训练批次
    """
    # 第一步：创建数据集
    # 从原始数据（如HuggingFace数据集）创建PyTorch数据集
    # 这一步会下载数据、解析数据格式、创建索引等
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    
    # 第二步：应用数据变换
    # 包括：键名映射、数据标准化、数据增强等
    # 把原始数据转换成模型能理解的格式
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # 第三步：处理分布式训练
    # 根据不同的框架和训练模式，设置不同的采样器和批次大小
    
    sampler = None  # 采样器，用于分布式训练
    
    if framework == "pytorch":
        # PyTorch框架：使用PyTorch的分布式训练
        if torch.distributed.is_initialized():
            # 多GPU分布式训练：创建DistributedSampler
            # 这个采样器确保每个GPU处理不同的数据子集
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),  # 总GPU数量
                rank=torch.distributed.get_rank(),                # 当前GPU的排名
                shuffle=shuffle,                                  # 是否打乱数据
                drop_last=True,                                   # 丢弃最后不完整的批次
            )
            # 每个GPU的本地批次大小 = 全局批次大小 / GPU数量
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            # 单GPU训练：使用全局批次大小
            local_batch_size = batch_size
    else:
        # JAX框架：使用JAX的分片机制
        # 每个进程的本地批次大小 = 全局批次大小 / 进程数量
        local_batch_size = batch_size // jax.process_count()

    # 记录本地批次大小，用于调试和监控
    logging.info(f"local_batch_size: {local_batch_size}")
    
    # 第四步：创建实际的数据加载器
    data_loader = TorchDataLoader(
        dataset,                                    # 处理后的数据集
        local_batch_size=local_batch_size,          # 本地批次大小
        sharding=None if framework == "pytorch" else sharding,  # 分片配置（仅JAX使用）
        shuffle=(sampler is None and shuffle),      # 是否打乱数据（如果使用采样器则不重复打乱）
        sampler=sampler,                            # 分布式采样器
        num_batches=num_batches,                    # 批次数量限制
        num_workers=num_workers,                    # 工作进程数
        seed=seed,                                  # 随机种子
        framework=framework,                        # 使用的框架
    )

    # 第五步：包装成统一的数据加载器接口
    # DataLoaderImpl提供了统一的接口，隐藏了不同实现的细节
    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """
    创建RLDS数据加载器
    
    创建基于RLDS（Robotics Learning Dataset）的数据加载器，专门用于机器人学习任务。
    RLDS是Google开发的一种机器人学习数据集格式，使用TensorFlow的TFRecord格式存储。
    
    这个函数的工作流程：
    1. 创建RLDS数据集（从TFRecord文件）
    2. 应用数据变换（标准化、键名映射等）
    3. 创建RLDS数据加载器
    4. 包装成统一接口
    
    注意：此数据加载器需要额外的依赖项 -- 参见 examples/droid/README_train.md
    
    Args:
        data_config: 数据配置，包含RLDS数据集路径和设置
        action_horizon: 动作序列长度，模型一次预测多少步动作
        batch_size: 批次大小，用于批处理数据
        sharding: JAX数据分片配置，用于多GPU训练
        skip_norm_stats: 是否跳过数据标准化（用于调试）
        shuffle: 是否打乱数据顺序，提高训练效果
        num_batches: 返回的批次数量限制，None表示使用整个数据集
        framework: 使用的框架（目前只支持"jax"）
        
    Returns:
        创建的RLDS数据加载器，可以迭代获取训练批次
        
    Raises:
        NotImplementedError: 如果使用PyTorch框架（暂不支持）
    """
    # 检查框架支持
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    
    # 第一步：创建RLDS数据集
    # 从TFRecord文件创建数据集，支持流式数据访问
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    
    # 第二步：应用数据变换
    # 对可迭代数据集应用变换，包括标准化、键名映射等
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    # 第三步：创建RLDS数据加载器
    # RLDSDataLoader专门处理RLDS格式的数据
    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,        # JAX分片配置
        num_batches=num_batches,  # 批次数量限制
    )

    # 第四步：包装成统一接口
    # DataLoaderImpl提供统一的接口，隐藏实现细节
    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """
    PyTorch数据加载器实现
    
    基于PyTorch的数据加载器，支持多进程数据加载、分布式训练和JAX分片。
    这是一个通用的数据加载器，可以同时支持PyTorch和JAX两种框架。
    
    主要特点：
    - 支持多进程数据加载（提高数据加载效率）
    - 支持分布式训练（PyTorch DDP和JAX分片）
    - 支持数据打乱和自定义采样器
    - 支持批次数量限制
    - 支持JAX分片机制
    
    使用场景：
    - 训练大规模数据集
    - 多GPU分布式训练
    - 需要高效数据加载的场景
    """

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """
        初始化PyTorch数据加载器
        
        Args:
            dataset: 要加载的数据集，必须实现Dataset协议
            local_batch_size: 每个进程的本地批次大小，会被分配到各个设备
            sharding: JAX数据分片配置，用于多GPU训练（仅JAX框架使用）
            shuffle: 是否打乱数据顺序，提高训练效果
            sampler: 自定义采样器，用于分布式训练（如DistributedSampler）
            num_batches: 返回的批次数量限制，None表示使用整个数据集
            num_workers: 数据加载工作进程数，0表示在主进程中执行
            seed: 随机种子，用于数据打乱和采样器初始化
            framework: 使用的框架（"jax" 或 "pytorch"）
            
        Raises:
            NotImplementedError: 如果使用多进程（暂不支持）
            ValueError: 如果批次大小超过数据集大小
        """
        # 检查多进程支持
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        # 验证批次大小
        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # 设置分片配置
        # PyTorch使用None，JAX使用分片配置
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # 为JAX创建默认的数据并行分片
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),  # 创建设备网格
                jax.sharding.PartitionSpec("B"),           # 按批次维度分片
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        # 创建PyTorch数据加载器
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),  # 类型转换
            batch_size=local_batch_size,                     # 批次大小
            shuffle=(sampler is None and shuffle),           # 是否打乱（如果使用采样器则不重复打乱）
            sampler=sampler,                                 # 自定义采样器
            num_workers=num_workers,                         # 工作进程数
            multiprocessing_context=mp_context,              # 多进程上下文
            persistent_workers=num_workers > 0,              # 持久化工作进程
            collate_fn=_collate_fn,                          # 批次整理函数
            worker_init_fn=_worker_init_fn,                  # 工作进程初始化函数
            drop_last=True,                                  # 丢弃最后不完整的批次
            generator=generator,                             # 随机数生成器
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        """
        迭代数据加载器，返回批次数据
        
        这个方法实现了数据加载器的迭代功能，支持：
        1. 批次数量限制
        2. 数据集循环遍历
        3. JAX分片和PyTorch张量转换
        """
        num_items = 0
        while True:
            # 创建数据迭代器
            data_iter = iter(self._data_loader)
            while True:
                # 检查批次数量限制
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    # 获取下一个批次
                    batch = next(data_iter)
                except StopIteration:
                    break  # 数据集已用完，创建新迭代器重新开始
                num_items += 1
                
                # 根据框架类型转换数据格式
                if self._sharding is not None:
                    # JAX框架：转换为分片数组
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    # PyTorch框架：转换为PyTorch张量
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """
    批次整理函数
    
    将多个数据项组合成一个批次，确保所有元素都转换为numpy数组后再堆叠。
    这个函数被PyTorch DataLoader使用，用于将多个样本组合成一个批次。
    
    Args:
        items: 多个数据项的列表
        
    Returns:
        堆叠后的批次数据
    """
    # 确保在堆叠之前转换为numpy数组，因为某些传入元素可能是JAX数组
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """
    工作进程初始化函数
    
    在多进程数据加载中，每个工作进程启动时会调用这个函数。
    主要作用是配置JAX环境，避免多进程间的GPU内存冲突。
    
    Args:
        worker_id: 工作进程的ID
        
    Note:
        这个方法在worker进程中导入jax之后调用，不能用于选择后端
    """
    # 设置JAX环境变量，避免多进程间的内存冲突
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # 不预分配GPU内存
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"  # 使用平台分配器


class RLDSDataLoader:
    """
    RLDS数据加载器
    
    RLDS（Robotics Learning Dataset）数据加载器，专门用于处理RLDS格式的数据集。
    这是DROID数据加载器的浅层包装器，使其与openpi兼容。
    
    主要特点：
    - 专门处理RLDS格式的数据集
    - 支持JAX分片机制
    - 所有批处理已经在DROID数据集中完成
    - 轻量级包装，不需要额外处理
    
    使用场景：
    - 处理DROID数据集
    - 机器人学习任务
    - 大规模机器人数据训练
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        """
        初始化RLDS数据加载器
        
        Args:
            dataset: DROID RLDS数据集，已经完成批处理
            sharding: JAX数据分片配置，用于多GPU训练
            num_batches: 返回的批次数量限制，None表示使用整个数据集
            
        Raises:
            NotImplementedError: 如果使用多进程（暂不支持）
        """
        self._dataset = dataset
        self._num_batches = num_batches

        # 检查多进程支持
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        # 设置分片配置
        if sharding is None:
            # 默认使用数据并行分片
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        """
        迭代RLDS数据加载器，返回批次数据
        
        这个方法实现了RLDS数据加载器的迭代功能，支持：
        1. 批次数量限制
        2. 数据集循环遍历
        3. JAX分片转换
        """
        num_items = 0
        while True:
            # 创建数据集迭代器
            data_iter = iter(self._dataset)
            while True:
                # 检查批次数量限制
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    # 获取下一个批次
                    batch = next(data_iter)
                except StopIteration:
                    break  # 数据集已用完，创建新迭代器重新开始
                num_items += 1
                
                # 将批次数据转换为JAX分片数组
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    """
    数据加载器实现类
    
    数据加载器接口的具体实现，将底层数据加载器包装成统一的接口。
    这是数据加载系统的最终输出层，负责：
    1. 将底层数据加载器包装成统一接口
    2. 将原始批次数据转换为模型期望的观察和动作格式
    3. 提供一致的迭代接口
    
    主要特点：
    - 统一的接口：隐藏不同数据加载器的实现细节
    - 格式转换：将原始数据转换为模型输入格式
    - 类型安全：确保返回正确的数据类型
    - 可迭代：支持for循环等迭代操作
    
    使用场景：
    - 训练循环中的数据迭代
    - 模型推理时的数据加载
    - 任何需要批次数据的场景
    """
    
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        """
        初始化数据加载器实现
        
        Args:
            data_config: 数据配置，包含数据格式和变换信息
            data_loader: 底层数据加载器（TorchDataLoader或RLDSDataLoader）
        """
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        """返回数据配置"""
        return self._data_config

    def __iter__(self):
        """
        迭代数据加载器，返回格式化的观察和动作数据
        
        这个方法实现了DataLoader接口，将底层数据加载器的原始批次数据
        转换为模型期望的观察和动作格式。
        
        Returns:
            tuple[_model.Observation, _model.Actions]: 观察数据和动作数据
        """
        for batch in self._data_loader:
            # 将原始批次数据转换为模型期望的格式
            # 1. 从字典中提取观察数据并转换为Observation对象
            # 2. 提取动作数据
            yield _model.Observation.from_dict(batch), batch["actions"]  # 这是一个生成器对象 也是迭代器对象 每次运行一次就暂停 通过next进行推进
