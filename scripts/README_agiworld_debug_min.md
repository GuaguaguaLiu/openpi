# AgiWorld G2 PI05 最小可复现说明

当前这版训练定义：

- 3 路图像输入：`top_head`、`hand_left`、`hand_right`
- `state` 只取 24 维核心状态
- `action` 只取 26 维目标动作
- 模型内部仍保持 PI05 预训练的 `32` 槽位
- `pi05_base` 会缓存到项目目录：`/home/rickyyzliu/code/openpi/.cache/openpi-assets/checkpoints/pi05_base/params`

## 1) 一次性准备

### 1.1 解压样例数据（`lite`）

```bash
cd /home/rickyyzliu/data/agibot-world/AgiBotWorld2026/simulation/take_laundry_detergent_to_cart/g2_swift_picker/lite
tar -xzf meta.tar.gz.000
tar -xzf data.tar.gz.000
tar -xzf videos.tar.gz.000
```

### 1.2 安装视频解码依赖（只做一次）

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

## 2) 维度定义

### 2.1 `state` 24 维

- `0, 1`: 左右夹爪位置
- `54..67`: 双臂关节角，共 14 维
- `96..98`: 头部关节位置，共 3 维
- `99..103`: 腰部关节位置，共 5 维

### 2.2 `action` 26 维

- `0, 1`: 左右夹爪动作
- `16..29`: 双臂关节位置动作，共 14 维
- `30..32`: 头部位置动作，共 3 维
- `33..37`: 腰部位置动作，共 5 维
- `38..39`: 底盘速度，共 2 维

### 2.3 模型内部维度

- `state` 输入到 PI05 前会补零到 `32` 维
- `action` 监督到 PI05 前会补零到 `32` 维
- 推理输出时只取前 `26` 维有效动作

## 3) 最终 debug 指令（PI05）

> 机器是 8 卡时，`batch_size` 要能被 8 整除。

```bash
cd /home/rickyyzliu/code/openpi
uv run python scripts/train_agiworld_pi05.py \
  --mode debug \
  --exp-name agiworld_pi05_g2_debug \
  --checkpoint-base-dir /tmp/openpi_agiworld_ckpt \
  --batch-size 8 \
  --num-workers 0 \
  --num-train-steps 10 \
  --log-interval 1 \
  --save-interval 100
```

成功标志：日志出现 `Step 0: ...`，并生成目录：

```bash
/tmp/openpi_agiworld_ckpt/agiworld_pi05/agiworld_pi05_g2_debug/0
```

## 4) 最终正式训练指令

```bash
cd /home/rickyyzliu/code/openpi
uv run python scripts/train_agiworld_pi05.py \
  --mode train \
  --exp-name agiworld_pi05_g2_train \
  --checkpoint-base-dir /home/rickyyzliu/code/openpi/checkpoints \
  --batch-size 32 \
  --num-workers 0 \
  --num-train-steps 20000 \
  --log-interval 100 \
  --save-interval 5000
```

## 5) 这两条命令的区别

- `debug`：只验训练链路，不加载正式基座权重，不开 wandb，步数小，日志更密
- `train`：加载项目内本地缓存的 `pi05_base`，8 卡 FSDP 正式训练，日志和保存频率更适合长训
- 当前 checkpoint 保留策略：`max_to_keep=1`
- 如果同时配置 `keep_period`，命中的里程碑 checkpoint 会额外保留

## 6) 关键文件

- `src/openpi/policies/agi_g2_policy.py`
- `src/openpi/training/config_agiworld_pi05.py`
- `scripts/train_agiworld_pi05.py`

