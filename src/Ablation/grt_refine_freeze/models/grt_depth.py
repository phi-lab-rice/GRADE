"""GRT occupancy-to-depth adapter used as GRADE's frozen Stage-1 model."""

import torch
import torch.nn as nn
from safetensors.torch import load_file

from checkpoints import require_checkpoint
from models.grt_augmentations import dequantize_depth, translate_radar
from models.grt_model import GRTSmall


class GRTDepth(nn.Module):
    """Run GRT and convert its 3D occupancy logits to discretized depth."""

    def __init__(self, grt_model=None):
        super().__init__()
        self.grt = grt_model if grt_model is not None else GRTSmall()

    def forward(self, radar):
        occupancy_logits = self.grt(translate_radar(radar))
        return dequantize_depth(occupancy_logits)


def load_grt_checkpoint(model, checkpoint_path, map_location="cpu"):
    """Load a weights-only GRT safetensors file with strict matching."""
    require_checkpoint(checkpoint_path, "GRT")
    model.grt.load_state_dict(load_file(checkpoint_path, device="cpu"), strict=True)
