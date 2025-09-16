# 如何用自己数据集来做训练

## 1. 在config.py里参考其他的trainconfig写一个自己的config (注意这里面的name，后面你run的时候就是根据这个name去加载拟定的config)

注意点(下面注释部分)：
```python
TrainConfig(
    name="agi_debug",  # 名字要注意 根据名字加载配置
    model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
    data=LeRobotAgiDataConfig(  # 这个是自己单独定义的
        repo_id="test_agi_rickyyzliu_0915",  # 注意: 从HF_LEROBOT_HOME/test_agi_rickyyzliu_0915 下面去加载数据
        base_config=DataConfig(prompt_from_task=True, action_sequence_keys=("action",)),  # 注意: action_sequence_keys=("action",) 这里的action要和你数据集里action名字保持一致，小心actions！
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
```

## 2. LeRobotAgiDataConfig的定义 (下面两个TODO的地方要注意)

```python
# TODO 注意生成数据的时候生成的task都是debug有问题
# TODO 需要注意是把右侧映射成左侧,并且映射完只剩下左侧的内容(不会保留原来数据集的内容)
# 其实右侧就是你建立lerobot数据集叫的名字
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
    inputs=[agi_policy.AgiInputs(model_type=model_config.model_type)],
    outputs=[agi_policy.AgiOutputs()],
)
```

## 3. agi_policy的定义(参考libero_policy.py)

```python
# 输入部分：
# TODO 注意右侧的key要和config里你定义的repack_transform配合着来（transform里左侧的key就是这里用的key）
base_image = _parse_image(data["cam_high"])  # 主摄像头图像
wrist_left_image = _parse_image(data["cam_left_wrist"])  # 腕部摄像头图像
wrist_right_image = _parse_image(data["cam_right_wrist"])  # 腕部摄像头图像


# 输出部分：这里维度和你数据集里的维度对齐
return {"actions": np.asarray(data["actions"][:, :16])}