#!/usr/bin/env python3
"""测试脚本：加载第一帧训练数据，用训练好的模型预测action并对比GT。

该脚本用于：
1. 加载训练好的模型权重
2. 获取训练数据的第一帧（不打乱）
3. 使用模型预测action
4. 对比预测的action和GT action
5. 计算loss值

使用方法：
    python scripts/test_single_prediction.py <config_name> <checkpoint_path> [--step <step_number>]

参数：
    config_name: 配置名称（如 "pi0_libero"）
    checkpoint_path: 检查点路径
    step: 检查点步数（可选，默认使用最新）
"""

import logging
import pathlib
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import tyro

import openpi.models.model as _model
import openpi.policies.policy_config as _policy_config
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_policy_from_checkpoint(
    config: _config.TrainConfig, 
    checkpoint_path: str
) -> Any:
    """从检查点创建策略。
    
    Args:
        config: 训练配置
        checkpoint_path: 检查点路径
        
    Returns:
        策略对象
    """
    logger.info(f"Creating policy from checkpoint: {checkpoint_path}")
    
    # 使用 policy_config 的 create_trained_policy 方法
    policy = _policy_config.create_trained_policy(
        train_config=config,
        checkpoint_dir=checkpoint_path
    )
    
    logger.info("Policy created successfully")
    return policy


def load_first_batch(config: _config.TrainConfig) -> tuple[Any, Any]:
    """加载第一帧训练数据（不打乱）。
    
    Args:
        config: 训练配置
        
    Returns:
        观察和动作的元组
    """
    logger.info("Loading first batch of training data (no shuffle)")
    
    # 创建数据加载器（不打乱数据）
    data_loader = _data_loader.create_data_loader(
        config,
        shuffle=False,  # 不打乱数据
        num_batches=1,  # 只加载一个批次
    )
    
    # 获取第一个批次
    data_iter = iter(data_loader)
    batch = next(data_iter)
    
    logger.info(f"Loaded batch with shape: {training_utils.array_tree_to_info(batch)}")
    
    return batch[0], batch[1]  # observation, actions


def predict_actions(
    policy: Any, 
    observation: dict
) -> np.ndarray:
    """使用策略预测动作。
    
    Args:
        policy: 策略对象
        observation: 观察数据字典
        
    Returns:
        预测的动作
    """
    logger.info("Predicting actions")
    
    # 使用策略的infer方法预测动作
    result = policy.infer(observation)
    predicted_actions = result['actions']
    
    logger.info(f"Predicted actions shape: {predicted_actions.shape}")
    
    return predicted_actions


def compute_loss_directly(
    policy: Any,
    observation: dict,
    gt_actions: np.ndarray
) -> float:
    """直接计算预测动作和真实动作的loss。
    
    Args:
        policy: 策略对象
        observation: 观察数据字典
        gt_actions: 真实动作
        
    Returns:
        loss值
    """
    logger.info("Computing loss")
    
    # 预测动作
    result = policy.infer(observation)
    predicted_actions = result['actions']
    
    # 计算MSE loss
    mse_loss = float(np.mean((predicted_actions - gt_actions) ** 2))
    logger.info(f"MSE loss: {mse_loss:.6f}")
    
    return mse_loss


def compare_actions(
    predicted_actions: _model.Actions, 
    gt_actions: _model.Actions
) -> dict[str, float]:
    """对比预测动作和真实动作。
    
    Args:
        predicted_actions: 预测的动作
        gt_actions: 真实动作
        
    Returns:
        对比统计信息
    """
    logger.info("Comparing predicted and ground truth actions")
    
    # 转换为numpy数组
    pred_np = np.array(predicted_actions)
    gt_np = np.array(gt_actions)
    
    # 计算各种统计指标
    mse = float(np.mean((pred_np - gt_np) ** 2))
    mae = float(np.mean(np.abs(pred_np - gt_np)))
    max_error = float(np.max(np.abs(pred_np - gt_np)))
    
    # 计算每个维度的误差
    dim_errors = np.mean(np.abs(pred_np - gt_np), axis=(0, 1))  # 平均每个动作维度的误差
    
    stats = {
        "mse": mse,
        "mae": mae,
        "max_error": max_error,
        "dim_errors": dim_errors.tolist(),
    }
    
    logger.info(f"Action comparison stats: {stats}")
    
    return stats


def main(
    config_name: str,
    checkpoint_path: str,
    output_dir: str | None = None
):
    """主函数：执行单帧预测测试。
    
    Args:
        config_name: 配置名称
        checkpoint_path: 检查点路径
        output_dir: 输出目录（可选）
    """
    logger.info(f"Starting single frame prediction test")
    logger.info(f"Config: {config_name}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    
    # 1. 获取配置
    config = _config.get_config(config_name)
    logger.info(f"Loaded config: {config.name}")
    
    # 2. 创建策略（包含模型加载）
    policy = create_policy_from_checkpoint(config, checkpoint_path)
    print("policy created!")

    
    # 3. 加载第一帧数据
    observation, gt_actions = load_first_batch(config)
    
    # 4. 将观察数据转换为策略期望的字典格式
    # 策略的infer方法期望原始的、未变换的数据格式
    # 根据agi_debug配置，策略期望的输入格式是：
    # cam_high, cam_left_wrist, cam_right_wrist, state, prompt
    # 注意：这里需要提供原始的图像数据，而不是经过变换的数据
    
    # 从训练数据中提取原始格式（这里需要根据实际数据结构调整）
    # 由于训练数据已经经过变换，我们需要构造策略期望的原始格式
    observation_dict = {
        "cam_high": observation.images["base_0_rgb"],  # 主摄像头图像
        "cam_left_wrist": observation.images["left_wrist_0_rgb"],  # 左腕摄像头图像
        "cam_right_wrist": observation.images["right_wrist_0_rgb"],  # 右腕摄像头图像
        "state": observation.state,
        "prompt": "debug task",  # 使用默认提示词，因为tokenized_prompt需要特殊处理
    }
    
    # 5. 预测动作
    predicted_actions = predict_actions(policy, observation_dict)
    
    # 6. 计算loss
    loss = compute_loss_directly(policy, observation_dict, np.array(gt_actions))
    
    # 7. 对比动作
    comparison_stats = compare_actions(predicted_actions, np.array(gt_actions))
    
    # 8. 打印结果
    print("\n" + "="*60)
    print("单帧预测测试结果")
    print("="*60)
    print(f"配置名称: {config_name}")
    print(f"检查点路径: {checkpoint_path}")
    print(f"Loss值: {loss:.6f}")
    print(f"MSE: {comparison_stats['mse']:.6f}")
    print(f"MAE: {comparison_stats['mae']:.6f}")
    print(f"最大误差: {comparison_stats['max_error']:.6f}")
    print(f"各维度平均误差: {comparison_stats['dim_errors']}")
    
    # 打印动作对比详情
    print("\n动作对比详情:")
    print(f"预测动作形状: {predicted_actions.shape}")
    print(f"真实动作形状: {np.array(gt_actions).shape}")
    print(f"预测动作 (前5个时间步):")
    print(predicted_actions[0, :5])
    print(f"真实动作 (前5个时间步):")
    print(np.array(gt_actions)[0, :5])
    print(f"动作差值 (前5个时间步):")
    print(predicted_actions[0, :5] - np.array(gt_actions)[0, :5])
    
    # 9. 保存结果（如果指定了输出目录）
    if output_dir:
        output_path = pathlib.Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存结果到文件
        results = {
            "config_name": config_name,
            "checkpoint_path": checkpoint_path,
            "loss": loss,
            "comparison_stats": comparison_stats,
            "predicted_actions": predicted_actions.tolist(),
            "gt_actions": np.array(gt_actions).tolist(),
        }
        
        import json
        with open(output_path / "prediction_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path / 'prediction_results.json'}")
    
    print("="*60)


if __name__ == "__main__":
    tyro.cli(main)
