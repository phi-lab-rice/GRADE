"""Nested YAML config loading, in the rcd_rice style.

One config.yaml with sections (data / depth / rcnet / sml / mono /
global_alignment / runtime / wandb) deep-merged over DEFAULT_CONFIG, with
attribute access (cfg.rcnet.epochs) and data paths resolved relative to the
config file.  The same module is shared by radarcam-depth and dataset_prep;
each folder's config.yaml only sets the sections it uses.

Environment overrides (applied after the YAML merge): RICE_DATA_ROOT ->
data.output_root, RICE_RAW_DIR -> data.raw_dir.
"""

import copy
import os
from typing import Any, Dict

import yaml

DEFAULT_CONFIG: Dict[str, Any] = {
    "data": {
        "raw_dir": "",
        "split_json": "split.json",
        "output_root": "rice_data",
        "smoke_eval_root": "",
        "limit": None,
    },
    "depth": {
        "max_radar_depth_m": 11.2,
        "min_radar_depth_m": 0.05,
        "min_pred_depth_m": 0.1,
        "max_pred_depth_m": 20.0,   # ZED GT can exceed the radar range
        "min_eval_depth_m": 0.0,
        "max_eval_depth_m": 11.2,
    },
    "rcnet": {
        # Shared-data convention: 288x512 images (uniform 0.4 scale of
        # 1280x720), patch 288x96 -> latent (9, 3) as in the ZJU config.
        "input_height": 288,
        "input_width": 512,
        "patch_size": [288, 96],
        "total_points_sampled": 40,
        "sample_probability_of_lidar": 0.10,
        "normalized_image_range": [0, 1],
        # Network (baseline architecture, unchanged)
        "encoder_type": ["rcnet", "batch_norm"],
        "n_filters_encoder_image": [32, 64, 128, 128, 128],
        "n_neurons_encoder_depth": [32, 64, 128, 128, 128],
        "decoder_type": ["multiscale", "batch_norm"],
        "n_filters_decoder": [256, 128, 64, 32, 16],
        "weight_initializer": "kaiming_uniform",
        "activation_func": "leaky_relu",
        # Augmentation (baseline defaults)
        "augmentation_probability": 1.0,
        "augmentation_random_brightness": [0.80, 1.20],
        "augmentation_random_contrast": [0.80, 1.20],
        "augmentation_random_saturation": [0.80, 1.20],
        "augmentation_random_flip_type": ["horizontal"],
        # Loss
        "w_positive_class": 2.5,
        "max_distance_correspondence": 0.5,
        "set_invalid_to_negative_class": False,
        # Training (paper Sec. IV-B: 50 epochs at lr 2e-4)
        "batch_size": 6,  # per GPU
        "epochs": 50,
        "learning_rate": 2e-4,
        "weight_decay": 0.0,
        "lr_milestones": [],
        "lr_gamma": 0.5,
        # Runtime
        "num_workers": 0,
        "log_freq": 50,
        "save_dir": "checkpoints_rcnet",
        "checkpoint_path": "",
        "response_thr": 0.5,
    },
    "sml": {
        "mono_tag": "dpt_hybrid",
        # Loss (w_lidar_loss MUST stay 0 for rice -- dense ZED gt)
        "loss_func": "smoothl1",
        "w_smoothness": 0.0,
        "loss_smoothness_kernel_size": -1,
        "w_lidar_loss": 0.0,
        # Training (paper: lr 2e-4 -> 5e-5 after 20 of 40 epochs)
        "batch_size": 8,  # per GPU
        "epochs": 40,
        "learning_rate": 2e-4,
        "lr_milestones": [20],
        "lr_gamma": 0.25,
        "weight_decay": 0.0,
        # Runtime
        "num_workers": 0,
        "log_freq": 50,
        "save_dir": "checkpoints_sml",
        "checkpoint_path": "",
        "save_visualizations": True,
        "num_visualizations": 4,
        "visualization_dir": "visualizations_sml",
    },
    "mono": {
        "model_type": "DPT_Hybrid",
        "tag": "dpt_hybrid",
    },
    "global_alignment": {
        "mono_tag": "dpt_hybrid",
        "min_points": 5,
    },
    "runtime": {
        "seed": 42,
        "cpu": False,
        "mixed_precision": "fp16",
    },
    "wandb": {
        "project": "radarcam-rice",
        "entity": None,
        "rcnet_run_name": "rcnet-run-1",
        "sml_run_name": "sml-run-1",
        "api_key": "",  # empty -> use `wandb login` / WANDB_API_KEY env
    },
}


class ConfigNode(dict):
    """Dictionary with attribute access for YAML config sections."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _to_node(value: Any) -> Any:
    if isinstance(value, dict):
        return ConfigNode({k: _to_node(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_node(v) for v in value]
    return value


def _resolve_paths(cfg: Dict[str, Any], config_path: str) -> None:
    config_dir = os.path.dirname(os.path.abspath(config_path))
    for key in ("raw_dir", "split_json", "output_root", "smoke_eval_root"):
        value = cfg["data"][key]
        if value and not os.path.isabs(value):
            cfg["data"][key] = os.path.abspath(os.path.join(config_dir, value))


def load_config(config_path: str) -> ConfigNode:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    with open(config_path, "r") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError("YAML config must contain a mapping at the top level.")
    _deep_update(cfg, payload)
    _resolve_paths(cfg, config_path)

    if os.environ.get("RICE_DATA_ROOT"):
        cfg["data"]["output_root"] = os.path.abspath(os.environ["RICE_DATA_ROOT"])
    if os.environ.get("RICE_RAW_DIR"):
        cfg["data"]["raw_dir"] = os.path.abspath(os.environ["RICE_RAW_DIR"])

    return _to_node(cfg)


def to_plain_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: to_plain_dict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_plain_dict(v) for v in value]
    return value


def loggable_config(cfg: Any) -> Dict[str, Any]:
    """Plain dict for wandb/checkpoint payloads, with the API key masked."""
    payload = to_plain_dict(cfg)
    if payload.get("wandb", {}).get("api_key"):
        payload["wandb"]["api_key"] = "***"
    return payload
