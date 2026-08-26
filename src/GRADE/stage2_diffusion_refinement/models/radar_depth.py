"""Compatibility export for the Stage-1 radar-depth model.

Stage-2 entry points historically imported ``models.radar_depth`` from their
own source folder.  The canonical implementation now lives in the explicit
``GRADE/stage1_radar_depth_module`` directory; re-exporting it here keeps those
released entry points and their safetensors state-dict keys unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path


STAGE1_DIR = Path(__file__).resolve().parents[2] / "stage1_radar_depth_module"
if str(STAGE1_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE1_DIR))

from radar_depth import RadarDepth  # noqa: E402


__all__ = ["RadarDepth"]
