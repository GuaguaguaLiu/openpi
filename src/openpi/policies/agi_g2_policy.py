import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


G2_STATE_INDICES = (
    0,
    1,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    66,
    67,
    96,
    97,
    98,
    99,
    100,
    101,
    102,
    103,
)
G2_ACTION_INDICES = (
    0,
    1,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
)


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _select_dims(values, indices: tuple[int, ...]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values[..., list(indices)]


@dataclasses.dataclass(frozen=True)
class AgiG2Inputs(transforms.DataTransformFn):
    model_type: _model.ModelType
    only_use_head_camera: bool = False

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["cam_high"])
        left_wrist_image = (
            np.zeros_like(base_image) if self.only_use_head_camera else _parse_image(data["cam_left_wrist"])
        )
        right_wrist_image = (
            np.zeros_like(base_image) if self.only_use_head_camera else _parse_image(data["cam_right_wrist"])
        )

        inputs = {
            "state": _select_dims(data["state"], G2_STATE_INDICES),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_ if self.only_use_head_camera else np.True_,
                "right_wrist_0_rgb": np.False_ if self.only_use_head_camera else np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = _select_dims(data["actions"], G2_ACTION_INDICES)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class AgiG2Outputs(transforms.DataTransformFn):
    action_dim: int = len(G2_ACTION_INDICES)

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
