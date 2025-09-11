# Pi0.5 使用指南

## 概述

Pi0.5是Pi0的升级版本，具有更好的开放世界泛化能力。本指南将帮助您快速上手Pi0.5模型。

## 环境安装

### 1. 安装uv包管理器

```bash
# 安装uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 重启终端或运行
source ~/.bashrc
```

### 2. 安装项目依赖

```bash
# 进入项目目录
cd /home/rickyyzliu/workspace/embodied-AI/manipulation/openpi

# 使用uv安装项目（自动创建虚拟环境）
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

**说明**：
- `uv sync` 会自动创建 `.venv` 虚拟环境并安装所有依赖
- 所有包都安装在项目目录的 `.venv/` 文件夹中
- 环境完全隔离，不会影响系统或其他项目

## 快速测试

### 方法1：使用简单客户端测试（推荐）

**步骤1：启动服务器**
```bash
# 终端1：启动Pi0.5-DROID模型服务器
uv run python scripts/serve_policy.py --env DROID --port 8000
```

**步骤2：运行测试客户端**
```bash
# 终端2：运行简单客户端测试
uv run python examples/simple_client/main.py --env DROID
```

**预期输出**：
- 服务器：`INFO:websockets.server:server listening on 0.0.0.0:8000`
- 客户端：显示推理速度统计表格

### 方法2：使用Docker测试

```bash
# 设置环境变量
export SERVER_ARGS="--env DROID"

# 使用Docker Compose一键测试
docker compose -f examples/simple_client/compose.yml up --build
```

### 方法3：直接Python代码测试

```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

# 加载Pi0.5-DROID模型
config = _config.get_config("pi05_droid")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")

# 创建策略
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# 创建测试观察
example = {
    "observation/exterior_image_1_left": "dummy_image_data",
    "observation/wrist_image_left": "dummy_image_data",
    "observation/joint_position": [0.0] * 7,
    "observation/gripper_position": [0.0],
    "prompt": "pick up the fork"
}

# 运行推理
action_chunk = policy.infer(example)["actions"]
print("生成的动作:", action_chunk)
```

### 其他测试选项

```bash
# 使用不同环境
uv run python scripts/serve_policy.py --env LIBERO --port 8000
uv run python examples/simple_client/main.py --env LIBERO

# 指定具体模型
uv run python scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid \
    --policy.dir=gs://openpi-assets/checkpoints/pi05_droid \
    --port 8000

# 启用记录功能
uv run python scripts/serve_policy.py --env DROID --record
```

## 可用的Pi0.5模型

| 模型 | 环境 | 配置名称 | 检查点路径 |
|------|------|----------|------------|
| Pi0.5-DROID | DROID | `pi05_droid` | `gs://openpi-assets/checkpoints/pi05_droid` |
| Pi0.5-LIBERO | LIBERO | `pi05_libero` | `gs://openpi-assets/checkpoints/pi05_libero` |
| Pi0.5-ALOHA | ALOHA | `pi05_aloha` | `gs://openpi-assets/checkpoints/pi05_base` |

## Pi0.5 vs Pi0 的主要区别

| 特性 | Pi0 | Pi0.5 |
|------|-----|-------|
| 状态输入 | 连续状态输入作为后缀 | 离散状态输入作为语言token |
| 时间建模 | 动作-时间MLP | adaRMSNorm时间注入 |
| Token长度 | 48 | 200 |
| 泛化能力 | 基础 | 更好的开放世界泛化 |

## 核心文件说明

我已经为以下核心文件添加了详细的中文注释：

### 1. `src/openpi/models/pi0_config.py` - Pi0.5配置类
- **功能**：定义Pi0和Pi0.5模型的所有配置参数
- **关键特性**：
  - `pi05: bool = False` - 启用Pi0.5模式的关键标志
  - `max_token_len` - Pi0.5使用200，Pi0使用48
  - `discrete_state_input` - Pi0.5使用离散状态输入

### 2. `src/openpi/models/pi0.py` - Pi0.5模型实现
- **功能**：Pi0/Pi0.5模型的核心实现
- **关键特性**：
  - `__init__()` - 根据pi05标志初始化不同的网络结构
  - `embed_prefix()` - 处理图像和语言输入
  - `embed_suffix()` - 处理动作和时间信息
  - `compute_loss()` - 计算训练损失
  - `sample_actions()` - 生成动作序列

### 3. `scripts/serve_policy.py` - 策略服务脚本
- **功能**：启动WebSocket服务器提供模型推理服务
- **关键特性**：
  - `DEFAULT_CHECKPOINT` - 各环境的默认模型配置
  - `create_policy()` - 根据参数创建策略
  - `main()` - 启动服务器主函数

## 测试说明

### 简单客户端测试原理

简单客户端会：
1. 连接到您的服务器（localhost:8000）
2. 生成随机的观察数据（图像、状态、提示）
3. 发送给Pi0.5模型进行推理
4. 接收生成的动作序列
5. 统计推理速度并显示结果

### 测试数据格式

DROID环境的测试数据包括：
- `observation/exterior_image_1_left`: 外部摄像头图像 (224x224x3)
- `observation/wrist_image_left`: 手腕摄像头图像 (224x224x3)  
- `observation/joint_position`: 关节位置 (7维)
- `observation/gripper_position`: 夹爪位置 (1维)
- `prompt`: 语言指令

## 故障排除

### 1. 内存不足
```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
uv run python scripts/serve_policy.py --env DROID
```

### 2. 端口被占用
```bash
uv run python scripts/serve_policy.py --env DROID --port 8001
```

### 3. 依赖问题
```bash
# 重新安装
rm -rf .venv
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

## 性能要求

| 模式 | 内存需求 | 示例GPU |
|------|----------|---------|
| 推理 | > 8 GB | RTX 4090 |
| 微调(LoRA) | > 22.5 GB | RTX 4090 |
| 微调(完整) | > 70 GB | A100 (80GB) / H100 |

## 下一步

测试成功后，您可以：

1. **集成到机器人系统**：参考 `examples/droid/README.md`
2. **微调模型**：参考 `examples/libero/README.md`
3. **开发自定义策略**：参考 `examples/aloha_real/README.md`
4. **远程推理**：参考 `docs/remote_inference.md`

## 总结

**完整的测试流程**：

```bash
# 1. 安装uv（如果还没有）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 安装项目
cd /path/to/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# 3. 启动Pi0.5-DROID服务器（终端1）
uv run python scripts/serve_policy.py --env DROID --port 8000

# 4. 运行测试客户端（终端2）
uv run python examples/simple_client/main.py --env DROID --num_steps 5
```

**预期结果**：
- 服务器启动：`INFO:websockets.server:server listening on 0.0.0.0:8000`
- 客户端测试：显示推理速度统计表格
- 推理速度：约80ms/次（在您的硬件上）

**测试成功标志**：
- ✅ 服务器成功启动并监听端口8000
- ✅ 客户端成功连接并发送测试数据
- ✅ Pi0.5模型成功生成动作序列
- ✅ 显示推理速度统计表格

所有核心文件都已添加详细的中文注释，您可以直接查看代码了解Pi0.5的实现细节。
