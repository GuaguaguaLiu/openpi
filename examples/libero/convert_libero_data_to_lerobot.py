"""
将数据集转换为 LeRobot 格式的最小示例脚本。

本脚本使用 Libero 数据集（存储在 RLDS 格式中）作为示例，但可以轻松修改以适配
任何其他自定义格式的数据。

使用方法：
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

如果要将数据集推送到 Hugging Face Hub，可以使用以下命令：
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

注意：运行此脚本需要安装 tensorflow_datasets：
`uv pip install tensorflow tensorflow_datasets`

您可以从 https://huggingface.co/datasets/openvla/modified_libero_rlds 下载原始 Libero 数据集
转换后的数据集将保存到 $HF_LEROBOT_HOME 目录中。
运行此转换脚本大约需要 30 分钟。
"""

import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro

# 输出数据集的名称，也用于 Hugging Face Hub
REPO_NAME = "your_hf_username/libero"

# 原始 Libero 数据集名称列表
# 为简化起见，我们将多个 Libero 数据集合并为一个训练数据集
RAW_DATASET_NAMES = [
    "libero_10_no_noops",      # Libero-10 数据集（无空操作）
    "libero_goal_no_noops",    # Libero-Goal 数据集（无空操作）
    "libero_object_no_noops",  # Libero-Object 数据集（无空操作）
    "libero_spatial_no_noops", # Libero-Spatial 数据集（无空操作）
]


def main(data_dir: str, *, push_to_hub: bool = False):
    """主函数：将 Libero 数据集转换为 LeRobot 格式。

    Args:
        data_dir: 原始 Libero 数据集所在的目录路径。
        push_to_hub: 是否将转换后的数据集推送到 Hugging Face Hub。
    """
    # 清理输出目录中任何现有的数据集
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # 创建 LeRobot 数据集，定义要存储的特征
    # OpenPi 假设本体感受信息存储在 `state` 中，动作存储在 `action` 中
    # LeRobot 假设图像数据的 dtype 为 `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,  # 数据集仓库 ID
        robot_type="panda",  # 机器人类型（Panda 机械臂）
        fps=10,  # 数据采集帧率（每秒 10 帧）
        features={
            # 主摄像头图像特征定义
            "image": {
                "dtype": "image",  # 图像数据类型
                "shape": (256, 256, 3),  # 图像尺寸：高度 x 宽度 x 通道数
                "names": ["height", "width", "channel"],  # 维度名称
            },
            # 腕部摄像头图像特征定义
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            # 机器人状态特征定义（本体感受信息）
            "state": {
                "dtype": "float32",  # 浮点数类型
                "shape": (8,),  # 8 维状态向量（关节位置、速度等）
                "names": ["state"],
            },
            # 机器人动作特征定义
            "actions": {
                "dtype": "float32",
                "shape": (7,),  # 7 维动作向量（7 个关节的控制指令）
                "names": ["actions"],
            },
        },
        image_writer_threads=10,  # 图像写入线程数（并行处理）
        image_writer_processes=5,  # 图像写入进程数（多进程处理）
    )

    # 遍历原始 Libero 数据集并将情节写入 LeRobot 数据集
    # 您可以修改此部分以适配自己的数据格式
    for raw_dataset_name in RAW_DATASET_NAMES:
        # 加载原始数据集（RLDS 格式）
        raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
        
        # 遍历数据集中的每个情节（episode）
        for episode in raw_dataset:
            # 遍历情节中的每个步骤（step）
            for step in episode["steps"].as_numpy_iterator():
                # 将步骤数据添加到 LeRobot 数据集中
                dataset.add_frame(
                    {
                        "image": step["observation"]["image"],  # 主摄像头图像
                        "wrist_image": step["observation"]["wrist_image"],  # 腕部摄像头图像
                        "state": step["observation"]["state"],  # 机器人状态
                        "actions": step["action"],  # 执行的动作
                        "task": step["language_instruction"].decode(),  # 语言指令（解码为字符串）
                    }
                )
            # 保存当前情节到数据集
            dataset.save_episode()

    # 可选：将数据集推送到 Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],  # 数据集标签
            private=False,  # 公开数据集
            push_videos=True,  # 推送视频数据
            license="apache-2.0",  # 开源许可证
        )


if __name__ == "__main__":
    # 使用 tyro 将函数转换为命令行接口
    tyro.cli(main)
