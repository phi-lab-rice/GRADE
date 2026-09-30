"""Direct Accelerate backend for the no-3D-loss complete-model ablation."""

from __future__ import annotations

import sys
from pathlib import Path


STAGE2_DIR = Path(__file__).resolve().parents[2] / "GRADE" / "stage2_diffusion_refinement"
sys.path.insert(0, str(STAGE2_DIR))

from inference import parse_stage_args, run_full  # noqa: E402


if __name__ == "__main__":
    run_full(parse_stage_args("Run the no-3D-loss GRADE model.").config)
