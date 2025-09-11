"""
简单客户端测试脚本

这个脚本用于测试Pi0/Pi0.5模型的推理性能，无需真实机器人。
主要功能：
1. 连接到策略服务器
2. 生成随机观察数据
3. 发送给模型进行推理
4. 统计推理速度并显示结果

使用方法：
uv run python examples/simple_client/main.py --env DROID --num_steps 20
"""

import dataclasses
import enum
import logging
import pathlib
import time

import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
import polars as pl
import rich
import tqdm
import tyro

logger = logging.getLogger(__name__)


class EnvMode(enum.Enum):
    """
    支持的环境模式
    
    定义了所有支持的机器人环境，每个环境都有对应的观察数据格式。
    """
    ALOHA = "aloha"          # ALOHA真实机器人环境
    ALOHA_SIM = "aloha_sim"  # ALOHA仿真环境
    DROID = "droid"          # DROID机器人环境
    LIBERO = "libero"        # LIBERO基准测试环境


@dataclasses.dataclass
class Args:
    """
    命令行参数配置
    
    定义了客户端测试所需的所有参数，包括服务器连接、测试步数等。
    """
    # 要连接的服务器主机地址
    host: str = "0.0.0.0"
    # 要连接的服务器端口。如果为None，服务器将使用默认端口
    port: int | None = 8000
    # 用于服务器的API密钥（可选）
    api_key: str | None = None
    # 运行策略的步数
    num_steps: int = 20
    # 保存时间统计到parquet文件的路径（例如：timing.parquet）
    timing_file: pathlib.Path | None = None
    # 运行策略的环境
    env: EnvMode = EnvMode.ALOHA_SIM


class TimingRecorder:
    """
    时间统计记录器
    
    用于记录和统计不同操作的时间测量，包括推理时间、服务器响应时间等。
    """

    def __init__(self) -> None:
        """
        初始化时间记录器
        
        创建一个字典来存储不同操作的时间测量数据。
        """
        self._timings: dict[str, list[float]] = {}

    def record(self, key: str, time_ms: float) -> None:
        """
        记录指定操作的时间测量
        
        Args:
            key: 操作名称（如"client_infer_ms"、"server_infer_ms"等）
            time_ms: 时间测量值（毫秒）
        """
        if key not in self._timings:
            self._timings[key] = []
        self._timings[key].append(time_ms)

    def get_stats(self, key: str) -> dict[str, float]:
        """
        获取指定操作的统计信息
        
        Args:
            key: 操作名称
            
        Returns:
            包含各种统计指标的字典（均值、标准差、百分位数等）
        """
        times = self._timings[key]
        return {
            "mean": float(np.mean(times)),      # 均值
            "std": float(np.std(times)),        # 标准差
            "p25": float(np.quantile(times, 0.25)),  # 25%分位数
            "p50": float(np.quantile(times, 0.50)),  # 50%分位数（中位数）
            "p75": float(np.quantile(times, 0.75)),  # 75%分位数
            "p90": float(np.quantile(times, 0.90)),  # 90%分位数
            "p95": float(np.quantile(times, 0.95)),  # 95%分位数
            "p99": float(np.quantile(times, 0.99)),  # 99%分位数
        }

    def print_all_stats(self) -> None:
        """
        以简洁格式打印所有操作的统计信息
        
        使用rich库创建一个美观的表格来显示各种时间统计指标。
        """
        # 创建表格对象
        table = rich.table.Table(
            title="[bold blue]时间统计[/bold blue]",
            show_header=True,
            header_style="bold white",
            border_style="blue",
            title_justify="center",
        )

        # 添加指标列，使用自定义样式
        table.add_column("指标", style="cyan", justify="left", no_wrap=True)

        # 添加统计列，使用一致的样式
        stat_columns = [
            ("均值", "yellow", "mean"),
            ("标准差", "yellow", "std"),
            ("P25", "magenta", "p25"),
            ("P50", "magenta", "p50"),
            ("P75", "magenta", "p75"),
            ("P90", "magenta", "p90"),
            ("P95", "magenta", "p95"),
            ("P99", "magenta", "p99"),
        ]

        # 为每个统计列添加列头
        for name, style, _ in stat_columns:
            table.add_column(name, justify="right", style=style, no_wrap=True)

        # 为每个指标添加行，包含格式化的值
        for key in sorted(self._timings.keys()):
            stats = self.get_stats(key)
            values = [f"{stats[key]:.1f}" for _, _, key in stat_columns]
            table.add_row(key, *values)

        # 使用自定义控制台设置打印表格
        console = rich.console.Console(width=None, highlight=True)
        console.print(table)

    def write_parquet(self, path: pathlib.Path) -> None:
        """
        将时间统计保存到parquet文件
        
        Args:
            path: 保存文件的路径
        """
        logger.info(f"正在将时间统计写入 {path}")
        frame = pl.DataFrame(self._timings)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)


def main(args: Args) -> None:
    """
    主函数：运行客户端测试
    
    Args:
        args: 命令行参数配置
    """
    # 根据环境选择对应的观察数据生成函数
    obs_fn = {
        EnvMode.ALOHA: _random_observation_aloha,
        EnvMode.ALOHA_SIM: _random_observation_aloha,
        EnvMode.DROID: _random_observation_droid,
        EnvMode.LIBERO: _random_observation_libero,
    }[args.env]

    # 创建WebSocket客户端策略
    policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
    )
    logger.info(f"服务器元数据: {policy.get_server_metadata()}")

    # 发送几个观察数据以确保模型已加载
    for _ in range(2):
        policy.infer(obs_fn())

    # 创建时间统计记录器
    timing_recorder = TimingRecorder()

    # 运行指定步数的推理测试
    for _ in tqdm.trange(args.num_steps, desc="运行策略"):
        inference_start = time.time()
        action = policy.infer(obs_fn())
        
        # 记录客户端推理时间
        timing_recorder.record("client_infer_ms", 1000 * (time.time() - inference_start))
        
        # 记录服务器时间统计
        for key, value in action.get("server_timing", {}).items():
            timing_recorder.record(f"server_{key}", value)
            
        # 记录策略时间统计
        for key, value in action.get("policy_timing", {}).items():
            timing_recorder.record(f"policy_{key}", value)

    # 打印所有统计信息
    timing_recorder.print_all_stats()

    # 如果指定了时间文件路径，则保存统计结果
    if args.timing_file is not None:
        timing_recorder.write_parquet(args.timing_file)


def _random_observation_aloha() -> dict:
    """
    生成ALOHA环境的随机观察数据
    
    Returns:
        包含ALOHA机器人观察数据的字典，包括：
        - state: 14维机器人状态向量
        - images: 4个摄像头的图像数据（高视角、低视角、左手腕、右手腕）
        - prompt: 语言指令
    """
    return {
        "state": np.ones((14,)),  # 14维状态向量（关节位置等）
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),      # 高视角摄像头
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),       # 低视角摄像头
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8), # 左手腕摄像头
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8), # 右手腕摄像头
        },
        "prompt": "do something",  # 语言指令
    }


def _random_observation_droid() -> dict:
    """
    生成DROID环境的随机观察数据
    
    Returns:
        包含DROID机器人观察数据的字典，包括：
        - observation/exterior_image_1_left: 外部摄像头图像
        - observation/wrist_image_left: 手腕摄像头图像
        - observation/joint_position: 7维关节位置
        - observation/gripper_position: 1维夹爪位置
        - prompt: 语言指令
    """
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),  # 外部摄像头图像
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),        # 手腕摄像头图像
        "observation/joint_position": np.random.rand(7),  # 7维关节位置（Franka机器人）
        "observation/gripper_position": np.random.rand(1),  # 1维夹爪位置
        "prompt": "do something",  # 语言指令
    }


def _random_observation_libero() -> dict:
    """
    生成LIBERO环境的随机观察数据
    
    Returns:
        包含LIBERO机器人观察数据的字典，包括：
        - observation/state: 8维机器人状态
        - observation/image: 主摄像头图像
        - observation/wrist_image: 手腕摄像头图像
        - prompt: 语言指令
    """
    return {
        "observation/state": np.random.rand(8),  # 8维机器人状态
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),      # 主摄像头图像
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8), # 手腕摄像头图像
        "prompt": "do something",  # 语言指令
    }


if __name__ == "__main__":
    # 设置日志级别并启动主函数
    logging.basicConfig(level=logging.INFO)
    # cli 将python类转换为命令行参数
    main(tyro.cli(Args))
