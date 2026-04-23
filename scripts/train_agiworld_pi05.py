"""Run AgiWorld G2 PI05 training with local project cache."""

import os
import pathlib

import tyro

from openpi.shared import download as download_utils
from openpi.training.config_agiworld_pi05 import AgiWorldPi05ConfigArgs, build_agiworld_pi05_config

HF_LEROBOT_HOME = (
    "/home/rickyyzliu/data/agibot-world/AgiBotWorld2026/simulation/take_laundry_detergent_to_cart/g2_swift_picker"
)
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PROJECT_CACHE_DIR = PROJECT_ROOT / ".cache"
PI05_BASE_REMOTE_PATH = "gs://openpi-assets/checkpoints/pi05_base/params"


def _load_train_module():
    """Load scripts/train.py as a module without packaging scripts/."""
    import importlib.util

    train_py = pathlib.Path(__file__).with_name("train.py")
    spec = importlib.util.spec_from_file_location("openpi_train_script", train_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load train module from {train_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ensure_local_pi05_base_checkpoint(mode: str) -> None:
    if mode.lower() == "debug":
        return
    download_utils.maybe_download(PI05_BASE_REMOTE_PATH)


def main(args: AgiWorldPi05ConfigArgs) -> None:
    # Keep dataset root explicit and local to this launcher.
    os.environ["HF_LEROBOT_HOME"] = HF_LEROBOT_HOME
    os.environ.setdefault("OPENPI_DATA_HOME", str(PROJECT_CACHE_DIR))
    # Allow debug on incomplete sample data without full norm stats.
    os.environ.setdefault("OPENPI_ALLOW_MISSING_NORM_STATS", "1")
    _ensure_local_pi05_base_checkpoint(args.mode)

    cfg = build_agiworld_pi05_config(args)
    train_module = _load_train_module()
    train_module.main(cfg)


if __name__ == "__main__":
    main(tyro.cli(AgiWorldPi05ConfigArgs))
