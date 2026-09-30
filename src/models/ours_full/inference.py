from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))
from inference_runtime import launch

HERE = Path(__file__).resolve().parent
launch(SRC / "GRADE/stage2_diffusion_refinement/inference_full.py", ("--config", str(HERE / "config.yaml")))
