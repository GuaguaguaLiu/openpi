"""
策略服务脚本

这个脚本用于启动一个WebSocket服务器来提供预训练模型的推理服务。
支持多种机器人环境（ALOHA、DROID、LIBERO等）和多种模型（Pi0、Pi0.5等）。

主要功能：
1. 加载预训练的策略模型
2. 启动WebSocket服务器
3. 接收观察数据并返回动作预测
4. 支持记录策略行为用于调试

使用方法：
python scripts/serve_policy.py --env ALOHA_SIM --port 8000
python scripts/serve_policy.py policy:checkpoint --policy.config=pi05_droid --policy.dir=gs://openpi-assets/checkpoints/pi05_droid
"""

import dataclasses
import enum
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """
    支持的环境模式
    
    定义了所有支持的机器人环境，每个环境都有对应的默认模型配置。
    """
    ALOHA = "aloha"          # ALOHA真实机器人环境
    ALOHA_SIM = "aloha_sim"  # ALOHA仿真环境
    DROID = "droid"          # DROID机器人环境
    LIBERO = "libero"        # LIBERO基准测试环境


@dataclasses.dataclass
class Checkpoint:
    """
    从训练好的检查点加载策略
    
    用于指定要加载的具体模型检查点，包括配置名称和检查点目录。
    """
    # 训练配置名称（例如："pi05_droid"）
    config: str
    # 检查点目录（例如："gs://openpi-assets/checkpoints/pi05_droid"）
    dir: str


@dataclasses.dataclass
class Default:
    """
    使用环境的默认策略
    
    当选择Default时，会根据指定的环境自动选择对应的默认模型。
    """


@dataclasses.dataclass
class Args:
    """
    策略服务脚本的参数
    
    定义了启动策略服务器所需的所有参数，包括环境选择、端口配置等。
    """
    # 要服务的环境。仅在服务默认策略时使用
    env: EnvMode = EnvMode.ALOHA_SIM

    # 默认提示词。如果数据中没有"prompt"键或模型没有默认提示词时使用
    default_prompt: str | None = None

    # 服务策略的端口号
    port: int = 8000
    # 是否记录策略行为用于调试
    record: bool = False

    # 指定如何加载策略。如果未提供，将使用环境的默认策略
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# 每个环境应该使用的默认检查点
# 这个字典定义了当使用Default策略时，每个环境对应的默认模型配置
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",  # 使用Pi0.5-ALOHA配置
        dir="gs://openpi-assets/checkpoints/pi05_base",  # Pi0.5基础模型
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",  # 使用Pi0-ALOHA仿真配置
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",  # Pi0-ALOHA仿真模型
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",  # 使用Pi0.5-DROID配置
        dir="gs://openpi-assets/checkpoints/pi05_droid",  # Pi0.5-DROID模型
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",  # 使用Pi0.5-LIBERO配置
        dir="gs://openpi-assets/checkpoints/pi05_libero",  # Pi0.5-LIBERO模型
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """
    为给定环境创建默认策略
    
    Args:
        env: 环境模式
        default_prompt: 默认提示词
        
    Returns:
        创建好的策略对象
        
    Raises:
        ValueError: 如果环境模式不支持
    """
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """
    根据给定参数创建策略
    
    Args:
        args: 脚本参数
        
    Returns:
        创建好的策略对象
    """
    match args.policy:
        case Checkpoint():
            # 从指定的检查点创建策略
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            # 使用环境的默认策略
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    """
    主函数：启动策略服务器
    
    Args:
        args: 脚本参数
    """
    # 创建策略
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # 如果需要记录策略行为
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    # 获取主机信息
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    # 创建并启动WebSocket服务器
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",  # 监听所有网络接口
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()  # 永久运行服务器


if __name__ == "__main__":
    # 设置日志级别并启动主函数
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
