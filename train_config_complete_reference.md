# TrainConfig 完整参数参考

## 🎯 什么是TrainConfig？

TrainConfig是训练配置的核心，包含了训练过程中需要的所有参数。就像做菜时的"食谱"，告诉你用什么材料、怎么处理、多长时间等。

## 📑 快速查找索引

| 参数名 | 类型 | 默认值 | 作用 | 位置 |
|--------|------|--------|------|------|
| `name` | str | - | 配置名称 | 基本信息 |
| `project_name` | str | "openpi" | 项目名称 | 基本信息 |
| `exp_name` | str | - | 实验名称 | 基本信息 |
| `model` | BaseModelConfig | Pi0Config | 模型配置 | 模型配置 |
| `weight_loader` | WeightLoader | NoOpWeightLoader | 权重加载器 | 权重加载 |
| `pytorch_weight_path` | str | None | PyTorch权重路径 | 权重加载 |
| `pytorch_training_precision` | str | "bfloat16" | PyTorch训练精度 | 权重加载 |
| `lr_schedule` | LRScheduleConfig | CosineDecaySchedule | 学习率调度器 | 训练参数 |
| `optimizer` | OptimizerConfig | AdamW | 优化器配置 | 训练参数 |
| `ema_decay` | float | 0.99 | EMA衰减率 | 训练参数 |
| `freeze_filter` | Filter | Nothing | 冻结参数过滤器 | 训练参数 |
| `data` | DataConfigFactory | FakeDataConfig | 数据配置 | 数据配置 |
| `assets_base_dir` | str | "./assets" | 资源基础目录 | 路径配置 |
| `checkpoint_base_dir` | str | "./checkpoints" | 检查点基础目录 | 路径配置 |
| `seed` | int | 42 | 随机种子 | 训练控制 |
| `batch_size` | int | 32 | 批次大小 | 训练控制 |
| `num_workers` | int | 2 | 数据加载工作进程数 | 训练控制 |
| `num_train_steps` | int | 30000 | 训练步数 | 训练控制 |
| `log_interval` | int | 100 | 日志记录间隔 | 日志和检查点 |
| `save_interval` | int | 1000 | 检查点保存间隔 | 日志和检查点 |
| `keep_period` | int | 5000 | 检查点保留周期 | 日志和检查点 |
| `overwrite` | bool | False | 是否覆盖现有检查点 | 日志和检查点 |
| `resume` | bool | False | 是否恢复训练 | 日志和检查点 |
| `wandb_enabled` | bool | True | 是否启用WandB | 监控和元数据 |
| `policy_metadata` | dict | None | 策略元数据 | 监控和元数据 |
| `fsdp_devices` | int | 1 | FSDP设备数量 | 设备配置 |

## 📋 所有参数完整列表

基于实际的TrainConfig类定义，这里是所有参数的完整列表：

### 1. 基本信息
```python
name: str                           # 配置名称（必须唯一）
project_name: str = "openpi"        # 项目名称
exp_name: str                       # 实验名称（必需）
```

### 2. 模型配置
```python
model: BaseModelConfig              # 模型配置（默认：Pi0Config）
```

### 3. 权重加载
```python
weight_loader: WeightLoader         # 权重加载器（默认：NoOpWeightLoader）
pytorch_weight_path: str | None     # PyTorch权重路径（默认：None）
pytorch_training_precision: str     # PyTorch训练精度（默认："bfloat16"）
```

### 4. 训练参数
```python
lr_schedule: LRScheduleConfig       # 学习率调度器（默认：CosineDecaySchedule）
optimizer: OptimizerConfig          # 优化器配置（默认：AdamW）
ema_decay: float | None = 0.99      # EMA衰减率
freeze_filter: Filter               # 冻结参数过滤器（默认：Nothing）
```

### 5. 数据配置
```python
data: DataConfigFactory             # 数据配置（默认：FakeDataConfig）
```

### 6. 路径配置
```python
assets_base_dir: str = "./assets"           # 资源基础目录（存放小文件）
checkpoint_base_dir: str = "./checkpoints"  # 检查点基础目录（存放模型权重）
```

**重要区别**：
- **资源目录** (`assets_dirs`): 存放标准化统计信息、配置文件等小文件
- **权重路径** (`weight_loader.params_path`): 存放模型权重，通常从云端下载

### 7. 训练控制
```python
seed: int = 42                      # 随机种子
batch_size: int = 32                # 批次大小
num_workers: int = 2                # 数据加载工作进程数
num_train_steps: int = 30_000       # 训练步数
```

### 8. 日志和检查点
```python
log_interval: int = 100             # 日志记录间隔
save_interval: int = 1000           # 检查点保存间隔
keep_period: int | None = 5000      # 检查点保留周期
overwrite: bool = False             # 是否覆盖现有检查点
resume: bool = False                # 是否恢复训练
```

### 9. 监控和元数据
```python
wandb_enabled: bool = True          # 是否启用WandB
policy_metadata: dict | None        # 策略元数据（默认：None）
```

### 10. 设备配置
```python
fsdp_devices: int = 1               # FSDP设备数量
```

## 🔧 计算属性（Properties）

这些属性不是直接设置的，而是根据其他参数计算得出的：

```python
@property
def assets_dirs(self) -> pathlib.Path:
    """资源文件目录：assets_base_dir / name
    存放：标准化统计信息、配置文件等小文件
    """
    return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

@property  
def checkpoint_dir(self) -> pathlib.Path:
    """检查点目录：checkpoint_base_dir / name / exp_name
    存放：模型权重、训练状态等大文件
    """
    return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()
```

## 📁 目录结构说明

```
项目根目录/
├── assets/                          # 资源基础目录
│   └── pi05_libero/                # 具体配置的资源目录
│       ├── norm_stats.json         # 数据标准化统计信息
│       └── config.json             # 配置文件
├── checkpoints/                     # 检查点基础目录
│   └── pi05_libero/                # 具体配置的检查点目录
│       └── my_experiment/          # 实验名称目录
│           └── 1000/               # 训练步数目录
│               ├── train_state/    # 训练状态
│               ├── params/         # 模型参数
│               └── assets/         # 检查点相关的资源文件
└── 云端存储/
    └── gs://openpi-assets/checkpoints/pi05_base/params  # 预训练权重
```

## 📊 实际配置示例解析

你提供的配置：
```python
TrainConfig(
    name='pi05_libero',                    # 配置名称
    project_name='openpi',                 # 项目名称
    exp_name='my_experiment',              # 实验名称
    
    # 模型配置
    model=Pi0Config(
        action_dim=32,                     # 动作维度
        action_horizon=10,                 # 动作序列长度
        max_token_len=200,                 # 最大token长度
        dtype='bfloat16',                  # 数据类型
        paligemma_variant='gemma_2b',      # 视觉-语言模型
        action_expert_variant='gemma_300m', # 动作专家模型
        pi05=True,                         # 使用π₀.₅模型
        discrete_state_input=False         # 离散状态输入
    ),
    
    # 权重加载
    weight_loader=CheckpointWeightLoader(
        params_path='gs://openpi-assets/checkpoints/pi05_base/params'
    ),
    pytorch_weight_path='/path/to/your/pytorch_weight_path',
    pytorch_training_precision='bfloat16',
    
    # 学习率调度
    lr_schedule=CosineDecaySchedule(
        warmup_steps=10000,        # 预热步数
        peak_lr=5e-05,            # 峰值学习率
        decay_steps=1000000,      # 衰减步数
        decay_lr=5e-05            # 衰减后学习率
    ),
    
    # 优化器
    optimizer=AdamW(
        b1=0.9,                    # 一阶动量
        b2=0.95,                   # 二阶动量
        eps=1e-08,                 # 数值稳定性
        weight_decay=1e-10,        # 权重衰减
        clip_gradient_norm=1.0     # 梯度裁剪
    ),
    
    # 训练参数
    ema_decay=0.999,               # EMA衰减率
    freeze_filter=Nothing(),       # 冻结参数过滤器
    seed=42,                       # 随机种子
    batch_size=256,                # 批次大小
    num_workers=2,                 # 数据加载进程数
    num_train_steps=30000,         # 训练步数
    
    # 日志和检查点
    log_interval=100,              # 日志间隔
    save_interval=1000,            # 保存间隔
    keep_period=5000,              # 保留周期
    overwrite=True,                # 覆盖现有
    resume=False,                  # 不恢复训练
    
    # 监控
    wandb_enabled=True,            # 启用WandB
    policy_metadata=None,          # 策略元数据
    
    # 设备
    fsdp_devices=1,                # FSDP设备数
    
    # 数据配置
    data=LeRobotLiberoDataConfig(
        repo_id='physical-intelligence/libero',
        assets=AssetsConfig(assets_dir=None, asset_id=None),
        base_config=DataConfig(...),
        extra_delta_transform=False
    ),
    
    # 路径配置
    assets_base_dir='./assets',           # 资源基础目录
    checkpoint_base_dir='./checkpoints',  # 检查点基础目录
)
```

## 🎯 计算得出的路径

基于你的配置，以下路径会被自动计算：

```python
# 资源目录（存放小文件）
assets_dirs = "./assets/pi05_libero"
# 内容：norm_stats.json, config.json 等

# 检查点目录（存放大文件）
checkpoint_dir = "./checkpoints/pi05_libero/my_experiment"  
# 内容：模型权重、训练状态等

# 预训练权重路径（从云端下载）
weight_loader.params_path = "gs://openpi-assets/checkpoints/pi05_base/params"
# 内容：预训练的模型权重
```

## 💡 关键理解点

1. **直接参数**：在配置中直接设置的参数
2. **计算属性**：根据其他参数自动计算的属性（如`assets_dirs`、`checkpoint_dir`）
3. **默认值**：大部分参数都有默认值，可以只设置必要的参数
4. **嵌套配置**：`model`、`data`等是复杂的嵌套配置对象

## 🔧 常用修改

```python
# 修改批次大小
batch_size=128  # 从256改为128

# 修改训练步数
num_train_steps=50000  # 从30000改为50000

# 修改学习率
lr_schedule=CosineDecaySchedule(peak_lr=1e-05)  # 降低学习率

# 启用多GPU
fsdp_devices=4  # 使用4个GPU

# 修改数据
data=LeRobotDROIDDataConfig(repo_id='your-droid-dataset')
```
