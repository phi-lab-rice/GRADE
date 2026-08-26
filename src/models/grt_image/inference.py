from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))
from inference_runtime import launch

HERE = Path(__file__).resolve().parent
launch(
    SRC / "Baselines/grt_image/inference.py",
    ("--config", str(HERE / "config.yaml"), "--checkpoint", str(HERE.parents[2] / "checkpoints/baselines/grt_image/grt_image.safetensors"), "--output_dir", str(HERE.parents[2] / "inference_results/grt_image")),
)
