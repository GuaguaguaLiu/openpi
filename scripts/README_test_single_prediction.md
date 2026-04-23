# 单帧预测测试脚本使用说明

## 🎯 功能说明

`test_single_prediction.py` 脚本用于测试训练好的模型在单帧数据上的预测性能。它会：

1. **加载训练好的模型权重**
2. **获取训练数据的第一帧**（不打乱数据）
3. **使用模型预测action**
4. **对比预测的action和GT action**
5. **计算loss值**

## 📋 使用方法

### 基本用法

```bash
python scripts/test_single_prediction.py <config_name> <checkpoint_path> [--output-dir <output_directory>]
```

### 参数说明

- `config_name`: 配置名称（如 "pi0_libero", "pi0_aloha" 等）
- `checkpoint_path`: 检查点路径（包含检查点文件的目录）
- `--output-dir`: 输出目录（可选，保存结果到JSON文件）

### 使用示例

#### 1. 测试 agi_debug 配置

```bash
# 基本测试
python scripts/test_single_prediction.py agi_debug ./checkpoints/agi_debug/my_experiment

# 保存结果到文件
python scripts/test_single_prediction.py agi_debug ./checkpoints/agi_debug/my_experiment --output-dir ./test_results
```

#### 2. 测试其他配置

```bash
# 测试 pi0_libero 配置
python scripts/test_single_prediction.py pi0_libero ./checkpoints/pi0_libero/my_experiment

# 测试 pi0_aloha 配置
python scripts/test_single_prediction.py pi0_aloha ./checkpoints/pi0_aloha/my_experiment
```

## 📊 输出结果

脚本会输出以下信息：

### 1. 控制台输出

```
============================================================
单帧预测测试结果
============================================================
配置名称: pi0_libero
检查点路径: ./checkpoints/pi0_libero/my_experiment
检查点步数: 1000
Loss值: 0.123456
MSE: 0.098765
MAE: 0.234567
最大误差: 0.456789
各维度平均误差: [0.1, 0.2, 0.15, ...]

动作对比详情:
预测动作形状: (1, 10, 32)
真实动作形状: (1, 10, 32)
预测动作 (前5个时间步):
[[0.1, 0.2, ...], [0.3, 0.4, ...], ...]
真实动作 (前5个时间步):
[[0.15, 0.25, ...], [0.35, 0.45, ...], ...]
动作差值 (前5个时间步):
[[-0.05, -0.05, ...], [-0.05, -0.05, ...], ...]
============================================================
```

### 2. JSON文件输出（如果指定了 --output-dir）

```json
{
  "config_name": "pi0_libero",
  "checkpoint_path": "./checkpoints/pi0_libero/my_experiment",
  "step": 1000,
  "loss": 0.123456,
  "comparison_stats": {
    "mse": 0.098765,
    "mae": 0.234567,
    "max_error": 0.456789,
    "dim_errors": [0.1, 0.2, 0.15, ...]
  },
  "predicted_actions": [[[0.1, 0.2, ...], ...]],
  "gt_actions": [[[0.15, 0.25, ...], ...]]
}
```

## 🔍 结果解读

### 关键指标

1. **Loss值**: 模型在单帧数据上的损失，越小越好
2. **MSE (均方误差)**: 预测动作和真实动作的均方误差
3. **MAE (平均绝对误差)**: 预测动作和真实动作的平均绝对误差
4. **最大误差**: 所有动作维度中的最大误差
5. **各维度平均误差**: 每个动作维度的平均误差

### 动作形状说明

- 形状: `(batch_size, action_horizon, action_dim)`
- `batch_size`: 批次大小（通常为1）
- `action_horizon`: 动作序列长度（如10）
- `action_dim`: 动作维度（如32）

## ⚠️ 注意事项

1. **检查点路径**: 确保检查点路径正确，包含训练好的模型权重
2. **配置名称**: 使用正确的配置名称，必须与训练时使用的配置一致
3. **数据加载**: 脚本会加载第一帧数据，不打乱顺序
4. **内存使用**: 确保有足够的内存加载模型和数据

## 🐛 常见问题

### 1. 检查点未找到

```
ValueError: No checkpoints found in ./checkpoints/pi0_libero/my_experiment
```

**解决方案**: 检查检查点路径是否正确，确保目录中存在检查点文件。

### 2. 配置名称错误

```
ValueError: Config 'wrong_config' not found. Did you mean 'pi0_libero'?
```

**解决方案**: 使用正确的配置名称，可以参考可用的配置列表。

### 3. 步数不存在

```
ValueError: Step 9999 not found. Available steps: [1000, 2000, 3000]
```

**解决方案**: 使用可用的检查点步数，或省略 `--step` 参数使用最新检查点。

## 🔧 自定义修改

如果需要修改脚本行为，可以：

1. **修改数据加载**: 在 `load_first_batch` 函数中修改数据加载逻辑
2. **修改评估指标**: 在 `compare_actions` 函数中添加更多统计指标
3. **修改输出格式**: 在 `main` 函数中修改结果输出格式
4. **添加可视化**: 可以添加动作对比的可视化图表

## 📈 扩展功能

可以考虑添加的功能：

1. **批量测试**: 测试多个数据帧而不是只测试第一帧
2. **可视化**: 生成动作对比的图表
3. **统计分析**: 更详细的统计分析
4. **不同检查点对比**: 对比不同训练步数的模型性能
