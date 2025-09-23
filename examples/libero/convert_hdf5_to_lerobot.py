"""
通用的HDF5到LeRobot转换脚本。

支持通过配置文件定义多个数据集路径和转换参数，避免硬编码。

使用方法：
uv run examples/libero/convert_hdf5_to_lerobot.py --config config.yaml

配置文件示例 (config.yaml):
```yaml
# 输出数据集配置
output:
  name: "jaka_pick_bottle"
  robot_type: "jaka"
  fps: 10
  push_to_hub: false

# 数据集路径列表
datasets:
  - path: "/path/to/dataset1"
    task_name: "pick_nearest_bottle_jaka_random_table"
  - path: "/path/to/dataset2" 
    task_name: "pick_nearest_bottle_jaka"

# 数据映射配置
mapping:
  images:
    cam_high: "observation.images.cam_high"
    cam_left_wrist: "observation.images.cam_left_wrist"
    cam_right_wrist: "observation.images.cam_right_wrist"
  state: "observation.state"
  action: "action"

# 数据格式配置
data_format:
  image_shape: [360, 640, 3]
  state_dim: 16
  action_dim: 16

# 处理配置
processing:
  image_writer_threads: 10
  image_writer_processes: 5
  max_episodes_per_dataset: null  # null表示处理所有episode
```
"""

import glob
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

import h5py
import numpy as np
import tyro
import yaml

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件。
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"加载配置文件: {config_path}")
    return config


def setup_environment(env_config: Dict[str, str]) -> None:
    """设置环境变量。
    
    Args:
        env_config: 环境变量配置字典
    """
    logger.info("🔧 设置环境变量:")
    
    for key, value in env_config.items():
        if value:  # 只设置非空值
            os.environ[key] = value
            logger.info(f"   - {key}: {value}")
        else:
            logger.info(f"   - {key}: 使用系统默认值")
    
    # 确保目录存在
    if 'HF_HOME' in env_config and env_config['HF_HOME']:
        hf_home = Path(env_config['HF_HOME'])
        hf_home.mkdir(parents=True, exist_ok=True)
        logger.info(f"   ✅ 创建HF_HOME目录: {hf_home}")
    
    if 'HF_LEROBOT_HOME' in env_config and env_config['HF_LEROBOT_HOME']:
        lerobot_home = Path(env_config['HF_LEROBOT_HOME'])
        lerobot_home.mkdir(parents=True, exist_ok=True)
        logger.info(f"   ✅ 创建HF_LEROBOT_HOME目录: {lerobot_home}")


def get_lerobot_imports():
    """动态导入LeRobot相关模块，确保环境变量已设置。
    
    Returns:
        LeRobotDataset和HF_LEROBOT_HOME
    """
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
    return HF_LEROBOT_HOME, LeRobotDataset


def decode_image(image_data: np.ndarray) -> np.ndarray:
    """解码图像数据。
    
    Args:
        image_data: 编码的图像数据 (uint8数组)
        
    Returns:
        解码后的图像数组 (H, W, C)
    """
    try:
        # 方法1: 使用PIL解码JPEG数据
        try:
            from PIL import Image
            import io
            image = Image.open(io.BytesIO(image_data))
            return np.array(image)
        except Exception as e:
            logger.warning(f"PIL解码失败: {e}")
        
        # 方法2: 使用cv2解码
        try:
            import cv2
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            if image is not None:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.warning(f"OpenCV解码失败: {e}")
            
        # 如果都失败了，返回占位图像
        logger.warning(f"无法解码图像数据，使用占位图像。数据大小: {len(image_data)}")
        return np.zeros((360, 640, 3), dtype=np.uint8)
        
    except Exception as e:
        logger.error(f"图像解码失败: {e}")
        return np.zeros((360, 640, 3), dtype=np.uint8)


def process_episode_file(
    episode_path: str, 
    task_name: str, 
    hdf5_mapping: Dict[str, Any],
    lerobot_mapping: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """处理单个episode文件。
    
    Args:
        episode_path: episode文件路径
        task_name: 任务名称
        hdf5_mapping: HDF5数据源映射配置
        lerobot_mapping: LeRobot输出映射配置
        
    Returns:
        处理后的数据列表
    """
    frames = []
    
    try:
        with h5py.File(episode_path, 'r') as f:
            # 获取数据长度
            num_steps = len(f[hdf5_mapping['action']])
            logger.info(f"处理episode {episode_path}，包含 {num_steps} 个时间步")
            
            for step_idx in range(num_steps):
                # 解码图像数据
                cam_high = decode_image(f[hdf5_mapping['images']['cam_high']][step_idx])
                cam_left_wrist = decode_image(f[hdf5_mapping['images']['cam_left_wrist']][step_idx])
                cam_right_wrist = decode_image(f[hdf5_mapping['images']['cam_right_wrist']][step_idx])
                
                # 获取状态和动作数据
                state = f[hdf5_mapping['state']][step_idx]
                action = f[hdf5_mapping['action']][step_idx]
                
                # 创建帧数据，使用配置中的映射
                frame = {
                    lerobot_mapping['images']['cam_high']: cam_high,
                    lerobot_mapping['images']['cam_left_wrist']: cam_left_wrist,
                    lerobot_mapping['images']['cam_right_wrist']: cam_right_wrist,
                    lerobot_mapping['state']: state,
                    lerobot_mapping['action']: action,
                    "task": task_name,
                }
                
                frames.append(frame)
                
    except Exception as e:
        logger.error(f"处理episode文件 {episode_path} 时出错: {e}")
        
    return frames


def create_dataset_features(data_format: Dict[str, Any]) -> Dict[str, Any]:
    """根据配置创建数据集特征定义。
    
    Args:
        data_format: 数据格式配置
        
    Returns:
        特征定义字典
    """
    image_shape = tuple(data_format['image_shape'])
    
    features = {
        # 图像特征
        "observation.images.cam_high": {
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        },
        "observation.images.cam_left_wrist": {
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        },
        "observation.images.cam_right_wrist": {
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        },
        # 状态特征
        "observation.state": {
            "dtype": "float32",
            "shape": (data_format['state_dim'],),
            "names": ["state"],
        },
        # 动作特征
        "action": {
            "dtype": "float32",
            "shape": (data_format['action_dim'],),
            "names": ["action"],
        },
    }
    
    return features


def verify_dataset(dataset_name: str, data_format: Dict[str, Any]) -> bool:
    """验证转换后的数据集。
    
    Args:
        dataset_name: 数据集名称
        data_format: 数据格式配置
        
    Returns:
        验证是否成功
    """
    logger.info(f"🔍 开始验证数据集: {dataset_name}")
    
    try:
        # 动态导入LeRobot模块
        HF_LEROBOT_HOME, LeRobotDataset = get_lerobot_imports()
        
        # 加载数据集
        dataset_path = HF_LEROBOT_HOME / dataset_name
        if not dataset_path.exists():
            logger.error(f"❌ 数据集路径不存在: {dataset_path}")
            return False
        
        # 使用LeRobotDataset加载数据集
        dataset = LeRobotDataset(dataset_name)
        
        # 基本信息
        total_samples = len(dataset)
        logger.info(f"📊 数据集基本信息:")
        logger.info(f"   - 总样本数: {total_samples}")
        logger.info(f"   - 数据集路径: {dataset_path}")
        
        # 检查episode文件
        episode_files = list(dataset_path.glob('**/episode_*.parquet'))
        logger.info(f"   - Episode文件数: {len(episode_files)}")
        
        # 计算总文件大小
        total_size = sum(f.stat().st_size for f in dataset_path.rglob('*') if f.is_file())
        logger.info(f"   - 总文件大小: {total_size / (1024**3):.2f} GB")
        
        # 验证数据格式
        if total_samples > 0:
            # 获取第一个样本进行验证
            sample = dataset[0]
            logger.info(f"🔬 数据格式验证:")
            
            # 验证图像数据
            expected_image_shape = tuple(data_format['image_shape'])
            for img_key in ['observation.images.cam_high', 'observation.images.cam_left_wrist', 'observation.images.cam_right_wrist']:
                if img_key in sample:
                    img_shape = sample[img_key].shape
                    if img_shape == expected_image_shape:
                        logger.info(f"   ✅ {img_key}: {img_shape}")
                    else:
                        logger.warning(f"   ⚠️  {img_key}: 期望 {expected_image_shape}, 实际 {img_shape}")
                else:
                    logger.error(f"   ❌ 缺少图像字段: {img_key}")
            
            # 验证状态数据
            if 'observation.state' in sample:
                state_shape = sample['observation.state'].shape
                expected_state_shape = (data_format['state_dim'],)
                if state_shape == expected_state_shape:
                    logger.info(f"   ✅ observation.state: {state_shape}")
                else:
                    logger.warning(f"   ⚠️  observation.state: 期望 {expected_state_shape}, 实际 {state_shape}")
            else:
                logger.error(f"   ❌ 缺少状态字段: observation.state")
            
            # 验证动作数据
            if 'action' in sample:
                action_shape = sample['action'].shape
                expected_action_shape = (data_format['action_dim'],)
                if action_shape == expected_action_shape:
                    logger.info(f"   ✅ action: {action_shape}")
                else:
                    logger.warning(f"   ⚠️  action: 期望 {expected_action_shape}, 实际 {action_shape}")
            else:
                logger.error(f"   ❌ 缺少动作字段: action")
            
            # 验证任务字段
            if 'task' in sample:
                task_value = sample['task']
                logger.info(f"   ✅ task: {task_value}")
            else:
                logger.error(f"   ❌ 缺少任务字段: task")
            
            # 数据范围检查
            logger.info(f"📈 数据范围检查:")
            if 'observation.state' in sample:
                state_data = sample['observation.state']
                logger.info(f"   - 状态数据范围: [{state_data.min():.3f}, {state_data.max():.3f}]")
            
            if 'action' in sample:
                action_data = sample['action']
                logger.info(f"   - 动作数据范围: [{action_data.min():.3f}, {action_data.max():.3f}]")
            
            # 图像数据检查
            for img_key in ['observation.images.cam_high', 'observation.images.cam_left_wrist', 'observation.images.cam_right_wrist']:
                if img_key in sample:
                    img_data = sample[img_key]
                    logger.info(f"   - {img_key} 范围: [{img_data.min()}, {img_data.max()}] (dtype: {img_data.dtype})")
        
        # 检查数据集元信息
        meta_info_path = dataset_path / "meta" / "info.json"
        if meta_info_path.exists():
            logger.info(f"   ✅ 元信息文件存在: {meta_info_path}")
        else:
            logger.warning(f"   ⚠️  元信息文件不存在: {meta_info_path}")
        
        logger.info(f"✅ 数据集验证完成!")
        return True
        
    except Exception as e:
        logger.error(f"❌ 数据集验证失败: {e}")
        return False


def main(config_path: str = "examples/libero/config_jaka.yaml"):
    """主函数：根据配置文件转换HDF5数据到LeRobot格式。
    
    Args:
        config_path: 配置文件路径，默认为 examples/libero/config_jaka.yaml
    """
    # 加载配置
    config = load_config(config_path)
    
    # 设置环境变量
    env_config = config.get('environment', {})
    if env_config:
        setup_environment(env_config)
    
    # 动态导入LeRobot模块（在环境变量设置后）
    HF_LEROBOT_HOME, LeRobotDataset = get_lerobot_imports()
    
    test_mode = config.get('test_mode', {'enabled': False})
    verify_mode = config.get('verify_mode', {'enabled': False})
    output_config = config['output']
    datasets_config = config['datasets']
    hdf5_mapping = config['hdf5_mapping']
    lerobot_mapping = config['lerobot_mapping']
    data_format = config['data_format']
    processing = config['processing']
    
    # 验证模式：只验证已转换的数据集
    if verify_mode.get('enabled', False):
        dataset_name = verify_mode.get('dataset_name', output_config['name'])
        logger.info(f"🔍 验证模式已启用，验证数据集: {dataset_name}")
        success = verify_dataset(dataset_name, data_format)
        if success:
            logger.info("✅ 验证完成，数据集格式正确！")
        else:
            logger.error("❌ 验证失败，数据集存在问题！")
        return
    
    logger.info(f"开始转换HDF5数据到LeRobot格式")
    logger.info(f"输出数据集: {output_config['name']}")
    logger.info(f"机器人类型: {output_config['robot_type']}")
    logger.info(f"数据集数量: {len(datasets_config)}")
    
    if test_mode.get('enabled', False):
        logger.info(f"🧪 测试模式已启用，每个数据集最多处理 {test_mode.get('max_episodes_per_dataset', 2)} 个episode")
        # 在测试模式下修改输出名称
        output_config['name'] = output_config['name'] + '_test'
    
    # 清理输出目录
    output_path = HF_LEROBOT_HOME / output_config['name']
    if output_path.exists():
        shutil.rmtree(output_path)
        logger.info(f"清理现有输出目录: {output_path}")

    # 创建LeRobot数据集
    features = create_dataset_features(data_format)
    
    dataset = LeRobotDataset.create(
        repo_id=output_config['name'],
        robot_type=output_config['robot_type'],
        fps=output_config['fps'],
        features=features,
        image_writer_threads=processing['image_writer_threads'],
        image_writer_processes=processing['image_writer_processes'],
    )

    # 处理每个数据集
    total_episodes = 0
    for dataset_idx, dataset_config in enumerate(datasets_config):
        dataset_path = dataset_config['path']
        task_name = dataset_config['task_name']
        
        logger.info(f"处理数据集 {dataset_idx + 1}/{len(datasets_config)}: {dataset_path}")
        logger.info(f"任务名称: {task_name}")
        
        # 获取episode文件
        episode_files = glob.glob(f"{dataset_path}/episode_*.hdf5")
        episode_files.sort()
        
        # 测试模式限制
        if test_mode.get('enabled', False):
            max_episodes = test_mode.get('max_episodes_per_dataset', 2)
            episode_files = episode_files[:max_episodes]
            logger.info(f"测试模式：限制处理 {max_episodes} 个episode")
        
        logger.info(f"找到 {len(episode_files)} 个episode文件")
        
        for i, episode_file in enumerate(episode_files):
            if i % 100 == 0:
                logger.info(f"处理进度: {i}/{len(episode_files)}")
                
            frames = process_episode_file(episode_file, task_name, hdf5_mapping, lerobot_mapping)
            
            for frame in frames:
                dataset.add_frame(frame)
            
            dataset.save_episode()
            total_episodes += 1
    
    logger.info("数据转换完成！")
    logger.info(f"总共处理了 {total_episodes} 个episode")
    
    # 验证转换后的数据集
    logger.info("🔍 开始验证转换后的数据集...")
    success = verify_dataset(output_config['name'], data_format)
    if not success:
        logger.error("❌ 数据集验证失败！")
        return
    
    # 可选：推送到Hugging Face Hub
    if output_config.get('push_to_hub', False):
        logger.info("推送到Hugging Face Hub...")
        dataset.push_to_hub(
            tags=output_config.get('tags', ["robotics", "hdf5"]),
            private=output_config.get('private', False),
            push_videos=output_config.get('push_videos', True),
            license=output_config.get('license', "apache-2.0"),
        )
        logger.info("推送完成！")
    
    logger.info(f"数据集已保存到: {output_path}")
    
    # 显示统计信息
    if output_path.exists():
        episode_files = list(output_path.glob('**/episode_*.parquet'))
        logger.info(f"转换后的episode数量: {len(episode_files)}")
        
        total_size = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file())
        logger.info(f"总文件大小: {total_size / (1024**3):.2f} GB")


if __name__ == "__main__":
    tyro.cli(main)
