import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_agi_example() -> dict:
    """创建 Libero 策略的随机输入示例。

    该函数生成一个符合 Libero 数据集格式的随机示例，用于测试和演示。
    包含机器人状态、主摄像头图像、腕部摄像头图像和语言指令。

    Returns:
        包含以下键的字典：
        - "observation/state": 8 维机器人状态向量（随机浮点数）
        - "observation/image": 主摄像头图像（224x224x3，随机像素值）
        - "observation/wrist_image": 腕部摄像头图像（224x224x3，随机像素值）
        - "prompt": 语言指令字符串
    """
    return {
        "observation/state": np.random.rand(8),  # 8 维状态向量（关节位置、速度等）
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),  # 主摄像头图像
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),  # 腕部摄像头图像
        "prompt": "do something",  # 语言指令
    }


def _parse_image(image) -> np.ndarray:
    """解析和标准化图像数据。

    该函数处理不同格式的图像数据，确保输出为标准的 uint8 格式 (H, W, C)：
    1. 将输入转换为 numpy 数组
    2. 如果是浮点数格式，转换为 0-255 的 uint8 格式
    3. 如果是 (C, H, W) 格式，重新排列为 (H, W, C) 格式

    Args:
        image: 输入图像数据（可能是各种格式）

    Returns:
        标准化后的图像数组，格式为 (H, W, C)，数据类型为 uint8
    """

    image = np.asarray(image)
    
    # 如果是浮点数类型，转换为 0-255 的 uint8 格式
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    
    # 如果是 (C, H, W) 格式，重新排列为 (H, W, C) 格式
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    
    return image


@dataclasses.dataclass(frozen=True)
class AgiInputs(transforms.DataTransformFn):
    """Libero 数据集的输入变换类。

    该类用于将输入数据转换为模型期望的格式，在训练和推理时都会使用。
    主要功能：
    1) 解析和标准化图像数据（主摄像头、腕部摄像头）；
    2) 构建模型输入字典，包含状态、图像、图像掩码等；
    3) 处理动作数据和语言指令。

    对于您自己的数据集，可以复制此类并根据下面的注释修改键名，
    以将数据集的正确元素传递给模型。

    Attributes:
        model_type: 确定将使用哪个模型类型（不要为您的数据集更改此参数）
    """

    # 确定将使用哪个模型类型
    # 不要为您的数据集更改此参数
    model_type: _model.ModelType
    only_use_head_camera: bool

    def __call__(self, data: dict) -> dict:
        """执行输入数据变换。

        Args:
            data: 原始输入数据字典，包含观测、动作、提示等信息

        Returns:
            变换后的输入字典，格式符合模型期望
        """
        # 解析图像数据为 uint8 (H,W,C) 格式
        # 因为 LeRobot 自动存储为 float32 (C,H,W)，在策略推理时会跳过此步骤
        # 为您的数据集保留此逻辑，但如果您的数据集将图像存储在不同的键中
        # 而不是 "observation/image" 或 "observation/wrist_image"，请在下面修改
        
        # Pi0 模型目前支持三种图像输入：一个第三人称视角和两个腕部视角（左右）
        # 如果您的数据集没有特定类型的图像（例如腕部图像），您可以在此处注释掉
        # 并用零数组替换，就像我们为下面的右腕图像所做的那样

        if self.only_use_head_camera:
            # print("*********** Train only use head camera!!!!!!!!!!! ***********")
            base_image = _parse_image(data["cam_high"])  # 主摄像头图像
            # wrist_left_image = _parse_image(data["cam_left_wrist"])  # 腕部摄像头图像
            # wrist_right_image = _parse_image(data["cam_right_wrist"])  # 腕部摄像头图像

            # 创建输入字典。不要更改下面字典中的键名
            inputs = {
                "state": data["state"],  # 机器人状态
                "image": {
                    "base_0_rgb": base_image,  # 主摄像头图像
                    "left_wrist_0_rgb": np.zeros_like(base_image),  # 左腕摄像头图像
                    "right_wrist_0_rgb": np.zeros_like(base_image),  # 右腕摄像头图像（用零填充）
                },
                "image_mask": {
                    "base_0_rgb": np.True_,  # 主摄像头图像掩码（有效）
                    "left_wrist_0_rgb": np.True_,  # 左腕摄像头图像掩码（有效）
                    "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
                },
            }
        else:
            # TODO 注意右侧的key要和 config里你定义的 repack_transform 配合着来（transform里左侧的key就是这里用的key）
            base_image = _parse_image(data["cam_high"])  # 主摄像头图像
            wrist_left_image = _parse_image(data["cam_left_wrist"])  # 腕部摄像头图像
            wrist_right_image = _parse_image(data["cam_right_wrist"])  # 腕部摄像头图像

            # 创建输入字典。不要更改下面字典中的键名
            inputs = {
                "state": data["state"],  # 机器人状态
                "image": {
                    "base_0_rgb": base_image,  # 主摄像头图像
                    "left_wrist_0_rgb": wrist_left_image,  # 左腕摄像头图像
                    "right_wrist_0_rgb": wrist_right_image,  # 右腕摄像头图像（用零填充）
                },
                "image_mask": {
                    "base_0_rgb": np.True_,  # 主摄像头图像掩码（有效）
                    "left_wrist_0_rgb": np.True_,  # 左腕摄像头图像掩码（有效）
                    # 我们只为 pi0 模型掩码填充图像，而不是 pi0-FAST
                    # 不要为您的数据集更改此逻辑
                    "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
                },
            }

        # 将动作填充到模型动作维度。为您的数据集保留此逻辑
        # 动作仅在训练期间可用
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # 将提示（即语言指令）传递给模型
        # 为您的数据集保留此逻辑（但如果指令不是存储在 "prompt" 中，
        # 请修改键名；输出字典始终需要具有键 "prompt"）
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class AgiOutputs(transforms.DataTransformFn):
    """Libero 数据集的输出变换类。

    该类用于将模型输出转换回数据集特定格式，仅用于推理。
    主要功能：从模型输出的填充动作中提取正确数量的动作维度。

    对于您自己的数据集，可以复制此类并根据下面的注释修改动作维度。
    """

    def __call__(self, data: dict) -> dict:
        """执行输出数据变换。

        Args:
            data: 模型输出数据字典，包含填充后的动作

        Returns:
            变换后的输出字典，包含正确维度的动作
        """
        # 只返回前 N 个动作
        # 由于我们上面将动作填充到模型动作维度，现在需要解析出返回字典中正确数量的动作
        # 对于 Libero，我们只返回前 7 个动作（其余的是填充）
        # 对于您自己的数据集，将 `7` 替换为您数据集的动作维度
        return {"actions": np.asarray(data["actions"][:, :16])}
