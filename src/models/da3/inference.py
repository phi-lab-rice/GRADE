from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))
from inference_runtime import launch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
launch(
    SRC / "Baselines/da3/inference.py",
    ("--data_root", str(ROOT / "evaluation_dataset/Smoke-Eval"), "--checkpoint", str(ROOT / "checkpoints/baselines/da3/da3metric-large.safetensors"), "--output_dir", str(ROOT / "inference_results/da3")),
)
