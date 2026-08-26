"""Direct Accelerate backend for the no-3D-loss complete-model ablation."""

from __future__ import annotations

import sys
from pathlib import Path


STAGE2_DIR = Path(__file__).resolve().parents[2] / "GRADE" / "stage2_diffusion_refinement"
sys.path.insert(0, str(STAGE2_DIR))

from inference import main as stage2_main  # noqa: E402


if __name__ == "__main__":
    stage2_main()
