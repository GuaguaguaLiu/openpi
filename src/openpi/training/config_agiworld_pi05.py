"""Dedicated AgiWorld PI05 training configs.

This file intentionally keeps only AgiWorld PI05-related configs.
"""

import dataclasses
import pathlib

from typing_extensions import override

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.policies.agi_g2_policy as agi_g2_policy
import openpi.training.config as base_config
import openpi.training.optimizer as optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
LOCAL_PI05_BASE_PARAMS = str(PROJECT_ROOT / ".cache" / "openpi-assets" / "checkpoints" / "pi05_base" / "params")


@dataclasses.dataclass(frozen=True)
class AgiWorldPi05ConfigArgs:
    """CLI arguments for AgiWorld PI05 config construction."""

    # "debug" is for fast pipeline validation, "train" is for long runs.
    mode: str = "debug"
    exp_name: str = "agiworld_pi05_debug"
    checkpoint_base_dir: str = "./checkpoints_agiworld"
    overwrite: bool = False
    resume: bool = False
    # Model/data scale knobs.
    batch_size: int = 8
    num_workers: int = 0
    num_train_steps: int = 2
    log_interval: int = 1
    save_interval: int = 1
    # Optional: local path for existing PyTorch weights.
    pytorch_weight_path: str | None = None


@dataclasses.dataclass(frozen=True)
class AgiWorldPi05DataConfig(base_config.DataConfigFactory):
    """AgiWorld-only data config with dedicated key mapping."""

    repo_id: str = "lite"
    extra_delta_transform: bool = False
    only_use_head_camera: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> base_config.DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "cam_high": "observation.images.top_head",
                        "cam_left_wrist": "observation.images.hand_left",
                        "cam_right_wrist": "observation.images.hand_right",
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[
                agi_g2_policy.AgiG2Inputs(
                    model_type=model_config.model_type,
                    only_use_head_camera=self.only_use_head_camera,
                )
            ],
            outputs=[agi_g2_policy.AgiG2Outputs()],
        )

        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=base_config.ModelTransformFactory()(model_config),
            action_sequence_keys=("action",),
            prompt_from_task=True,
            # keep debug/training offline-friendly when sample norm stats are absent
            norm_stats={},
        )


def build_agiworld_pi05_config(args: AgiWorldPi05ConfigArgs) -> base_config.TrainConfig:
    """Build a clean AgiWorld PI05 TrainConfig."""

    is_debug = args.mode.lower() == "debug"
    return base_config.TrainConfig(
        name="agiworld_pi05",
        exp_name=args.exp_name,
        checkpoint_base_dir=args.checkpoint_base_dir,
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            action_dim=32,
            discrete_state_input=True,
        ),
        data=AgiWorldPi05DataConfig(
            repo_id="lite",
            extra_delta_transform=False,
            only_use_head_camera=False,
        ),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_train_steps=args.num_train_steps,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        fsdp_devices=8,
        # Debug mode avoids downloading large base checkpoints.
        weight_loader=(
            weight_loaders.NoOpWeightLoader()
            if is_debug
            else weight_loaders.CheckpointWeightLoader(LOCAL_PI05_BASE_PARAMS)
        ),
        pytorch_weight_path=args.pytorch_weight_path,
        wandb_enabled=not is_debug,
        overwrite=args.overwrite or is_debug,
        resume=args.resume,
    )
