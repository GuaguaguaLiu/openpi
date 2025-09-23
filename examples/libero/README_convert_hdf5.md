# 通用HDF5到LeRobot转换脚本

## 🎯 功能说明

`convert_hdf5_to_lerobot.py` 是一个通用的HDF5到LeRobot转换脚本，支持通过配置文件定义多个数据集路径和转换参数，避免硬编码。

## 📋 特性

- ✅ **配置文件驱动**: 所有参数通过YAML配置文件管理
- ✅ **测试模式**: 支持测试模式，可限制转换的episode数量
- ✅ **验证模式**: 支持验证已转换的数据集，检查数据格式和完整性
- ✅ **多数据集支持**: 支持处理多个数据集目录
- ✅ **灵活映射**: 可配置HDF5和LeRobot数据字段映射关系
- ✅ **批量处理**: 自动处理目录中的所有episode文件
- ✅ **图像解码**: 自动解码JPEG格式图像
- ✅ **错误处理**: 单个文件错误不影响整体转换
- ✅ **进度显示**: 实时显示转换进度
- ✅ **HF Hub支持**: 可选推送到Hugging Face Hub

## 🚀 使用方法

### 基本用法

```bash
# 使用默认配置文件
uv run examples/libero/convert_hdf5_to_lerobot.py

# 使用自定义配置文件
uv run examples/libero/convert_hdf5_to_lerobot.py --config-path config.yaml
```

### 配置文件示例

创建 `config.yaml` 文件：

```yaml
# 环境变量配置
environment:
  HF_HOME: "/home/tione/notebook/workspace/rickyyzliu/huggingface/"  # Hugging Face缓存目录
  HF_LEROBOT_HOME: "/home/tione/notebook/workspace/rickyyzliu/huggingface/lerobot/"  # LeRobot数据集存储目录

# 测试模式配置
test_mode:
  enabled: false  # 设置为true启用测试模式
  max_episodes_per_dataset: 2  # 测试模式下每个数据集最多处理的episode数量

# 验证模式配置
verify_mode:
  enabled: false  # 设置为true启用验证模式，只验证已转换的数据集
  dataset_name: "jaka_pick_bottle"  # 要验证的数据集名称

# 输出数据集配置
output:
  name: "jaka_pick_bottle"
  robot_type: "jaka"
  fps: 10
  push_to_hub: false
  tags: ["jaka", "robotics", "pick_bottle", "hdf5"]
  private: false
  push_videos: true
  license: "apache-2.0"

# 数据集路径列表
datasets:
  - path: "/path/to/dataset1"
    task_name: "pick_nearest_bottle_jaka_random_table"
  - path: "/path/to/dataset2" 
    task_name: "pick_nearest_bottle_jaka"

# HDF5数据源映射配置（从HDF5文件读取的键名）
hdf5_mapping:
  images:
    cam_high: "observations/images/cam_high"
    cam_left_wrist: "observations/images/cam_left_wrist"
    cam_right_wrist: "observations/images/cam_right_wrist"
  state: "observations/qpos"
  action: "action"

# LeRobot输出映射配置（输出到LeRobot数据集的键名）
lerobot_mapping:
  images:
    cam_high: "observation.images.cam_high"
    cam_left_wrist: "observation.images.cam_left_wrist"
    cam_right_wrist: "observation.images.cam_right_wrist"
  state: "observation.state"
  action: "action"

# 数据格式配置
data_format:
  image_shape: [360, 640, 3]  # [height, width, channels]
  state_dim: 16
  action_dim: 16

# 处理配置
processing:
  image_writer_threads: 10
  image_writer_processes: 5
```

## 📁 配置文件说明

### environment 配置
- `HF_HOME`: Hugging Face缓存目录，用于存储模型和数据集缓存
- `HF_LEROBOT_HOME`: LeRobot数据集存储目录，转换后的数据集将保存在此目录下

### output 配置
- `name`: 输出数据集名称
- `robot_type`: 机器人类型
- `fps`: 数据采集帧率
- `push_to_hub`: 是否推送到Hugging Face Hub
- `tags`: 数据集标签
- `private`: 是否私有数据集
- `push_videos`: 是否推送视频
- `license`: 开源许可证

### datasets 配置
- `path`: 数据集目录路径
- `task_name`: 任务名称标识
- `max_episodes`: 最大处理episode数量（null表示处理所有）

### mapping 配置
定义HDF5字段到LeRobot字段的映射关系：
- `images`: 图像字段映射
- `state`: 状态字段映射
- `action`: 动作字段映射

### data_format 配置
- `image_shape`: 图像尺寸 [高度, 宽度, 通道数]
- `state_dim`: 状态维度
- `action_dim`: 动作维度

### processing 配置
- `image_writer_threads`: 图像写入线程数
- `image_writer_processes`: 图像写入进程数
- `max_episodes_per_dataset`: 全局episode数量限制

## 🔧 使用示例

### 1. 快速开始（使用默认配置）

```bash
# 直接运行，使用默认配置文件 examples/libero/config_jaka.yaml
uv run examples/libero/convert_hdf5_to_lerobot.py
```

### 2. 测试转换（少量数据）

```bash
# 启用测试模式，只转换少量episode
# 在配置文件中设置 test_mode.enabled: true
uv run examples/libero/convert_hdf5_to_lerobot.py
```

### 3. 完整转换

```bash
# 完整转换所有数据
# 在配置文件中设置 test_mode.enabled: false
uv run examples/libero/convert_hdf5_to_lerobot.py
```

### 4. 验证数据集

```bash
# 只验证已转换的数据集，不进行转换
# 在配置文件中设置 verify_mode.enabled: true
uv run examples/libero/convert_hdf5_to_lerobot.py
```

### 5. 自定义配置

```bash
# 使用自定义配置文件
uv run examples/libero/convert_hdf5_to_lerobot.py --config-path my_config.yaml
```

## 📊 输出结果

转换完成后，数据集将保存在 `$HF_LEROBOT_HOME/<output_name>/` 目录中，包含：

- `data/chunk-*/episode_*.parquet`: 每个episode的数据文件
- `meta/info.json`: 数据集元信息
- `meta/episodes.jsonl`: episode信息
- `meta/tasks.jsonl`: 任务信息

## 🔧 自定义修改

### 添加新的数据集

在配置文件的 `datasets` 部分添加：

```yaml
datasets:
  - path: "/path/to/new/dataset"
    task_name: "new_task_name"
    max_episodes: 50
```

### 修改数据映射

在配置文件的 `mapping` 部分修改字段映射：

```yaml
mapping:
  images:
    cam_high: "observation.images.cam_high"
    # 添加新的图像字段
    cam_new: "observation.images.cam_new"
  state: "observation.state"
  action: "action"
```

### 修改数据格式

在配置文件的 `data_format` 部分修改数据格式：

```yaml
data_format:
  image_shape: [480, 640, 3]  # 修改图像尺寸
  state_dim: 32               # 修改状态维度
  action_dim: 32              # 修改动作维度
```

## ⚠️ 注意事项

1. **配置文件格式**: 确保YAML格式正确，注意缩进
2. **路径检查**: 确保数据集路径存在且可访问
3. **内存使用**: 大量数据可能需要较多内存
4. **磁盘空间**: 确保有足够空间存储转换后的数据
5. **图像格式**: 脚本假设图像是JPEG格式，其他格式可能需要修改解码逻辑

## 🐛 常见问题

### 1. 配置文件格式错误

```
yaml.scanner.ScannerError: while scanning for the next token
```

**解决方案**: 检查YAML格式，确保缩进正确（使用空格，不要使用制表符）

### 2. 数据集路径不存在

```
FileNotFoundError: [Errno 2] No such file or directory
```

**解决方案**: 检查配置文件中的路径是否正确

### 3. 图像解码失败

```
无法解码图像数据，使用占位图像
```

**解决方案**: 检查图像数据格式，可能需要修改 `decode_image` 函数

## 🎉 完成后的使用

转换完成后，可以使用LeRobot加载数据集：

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# 加载转换后的数据集
dataset = LeRobotDataset("jaka_pick_bottle")
print(f"数据集包含 {len(dataset)} 个样本")
```
