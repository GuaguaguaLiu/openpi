"""计算配置的归一化统计信息。

该脚本用于计算给定配置的归一化统计信息。它将计算数据集中数据的均值和标准差，
并将其保存到配置的资源文件目录中。

使用方法：
    python scripts/compute_norm_stats.py <config_name> [--max_frames <int>]

参数：
    config_name: 配置名称（如 "pi0_libero"）
    max_frames: 可选的最大帧数限制，用于快速计算统计信息
"""

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    """移除字符串类型数据的变换类。

    该类用于从数据字典中移除字符串类型的值，因为 JAX 不支持字符串类型，
    且在计算归一化统计信息时不需要字符串数据。
    """

    def __call__(self, x: dict) -> dict:
        """移除字符串类型的数据。

        Args:
            x: 输入数据字典。

        Returns:
            移除字符串后的数据字典。
        """
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    """创建 PyTorch 数据加载器。

    Args:
        data_config: 数据配置对象。
        action_horizon: 动作序列长度。
        batch_size: 批次大小。
        model_config: 模型配置对象。
        num_workers: 数据加载工作进程数。
        max_frames: 可选的最大帧数限制。

    Returns:
        数据加载器和批次数的元组。

    Raises:
        ValueError: 如果数据配置没有 repo_id。
    """
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    
    # 创建 PyTorch 数据集
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    
    # 应用数据变换
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,  # 重新打包变换
            *data_config.data_transforms.inputs,    # 数据变换
            # 移除字符串，因为 JAX 不支持字符串，且计算归一化统计信息时不需要
            RemoveStrings(),
        ],
    )
    
    # 确定批次数和是否打乱数据
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True  # 限制帧数时打乱数据以获得更好的统计信息
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False  # 使用全部数据时不需要打乱
    
    # 创建数据加载器
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    """创建 RLDS 数据加载器。

    Args:
        data_config: 数据配置对象。
        action_horizon: 动作序列长度。
        batch_size: 批次大小。
        max_frames: 可选的最大帧数限制。

    Returns:
        数据加载器和批次数的元组。
    """
    # 创建 RLDS 数据集
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    
    # 应用数据变换
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,  # 重新打包变换
            *data_config.data_transforms.inputs,    # 数据变换
            # 移除字符串，因为 JAX 不支持字符串，且计算归一化统计信息时不需要
            RemoveStrings(),
        ],
        is_batched=True,
    )
    
    # 确定批次数
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # 注意：此长度目前为 DROID 硬编码
        num_batches = len(dataset) // batch_size
    
    # 创建 RLDS 数据加载器
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(config_name: str, max_frames: int | None = None):
    """主函数：计算归一化统计信息。

    Args:
        config_name: 配置名称。
        max_frames: 可选的最大帧数限制，用于快速计算统计信息。
    """
    # 获取配置并创建数据配置
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    # 根据数据配置类型创建相应的数据加载器
    if data_config.rlds_data_dir is not None:
        # 使用 RLDS 数据加载器（主要用于 DROID 数据集）
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        # 使用 PyTorch 数据加载器
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    # 初始化统计信息收集器
    keys = ["state", "actions"]  # 需要计算统计信息的键
    stats = {key: normalize.RunningStats() for key in keys}

    # 遍历数据批次并更新统计信息
    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            # 更新每个键的运行统计信息
            stats[key].update(np.asarray(batch[key]))

    # 获取最终的归一化统计信息
    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    # 保存统计信息到文件
    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    # 使用 tyro 创建命令行接口
    tyro.cli(main)
