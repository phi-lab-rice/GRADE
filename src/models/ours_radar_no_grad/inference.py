from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))
from inference_runtime import launch

HERE = Path(__file__).resolve().parent
launch(SRC / "Ablation/ours_radar_no_grad/inference.py", ("--config", str(HERE / "config.yaml")))
