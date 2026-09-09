from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))
from inference_runtime import launch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
launch(
    SRC / "Baselines/radarcam-depth/smoke_eval_inference.py",
    ("--config", str(HERE / "config.yaml"), "--rcnet_checkpoint", str(ROOT / "checkpoints/baselines/radarcam-depth/radarcam-depth_rcnet.safetensors"), "--sml_checkpoint", str(ROOT / "checkpoints/baselines/radarcam-depth/radarcam-depth_sml.safetensors"), "--output_dir", str(ROOT / "inference_results/radarcam-depth")),
)
