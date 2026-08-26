"""Checkpoint path validation shared by training and inference entry points."""

import os


def require_checkpoint(checkpoint_path, label):
    """Return an existing checkpoint path or raise a labeled error."""
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"{label} checkpoint not found: {checkpoint_path}")
    return checkpoint_path
