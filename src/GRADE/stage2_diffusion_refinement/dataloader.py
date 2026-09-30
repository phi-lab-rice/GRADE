import json
import re
from typing import Optional, List, Dict, Tuple, Literal
from pathlib import Path
from torch.utils.data import DataLoader
import torch

from rice_dataset import RiceDataset


_HELD_OUT_BUILDING_PATTERN = re.compile(r"keck|dell", re.IGNORECASE)


def _load_split_config() -> Dict[str, List[str]]:
    """Load split configuration from split.json."""
    split_file = Path(__file__).parent / "split.json"
    if not split_file.exists():
        raise FileNotFoundError(f"Split configuration not found: {split_file}")

    with open(split_file, "r") as f:
        split_config = json.load(f)

    return split_config


def create_dataloader(
    rice_root_dir: Optional[str] = None,
    smoke_eval_root_dir: Optional[str] = None,
    split: Literal["train", "val", "test"] = "train",
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    frame_skip: int = 1,
    num_frames: int = 1,
    drop_last: bool = False,
    seed: int = 42,
    # Processing parameters
    scale_factor: float = 0.001,
    max_depth_m: float = 11.2,
    depth_resolution: Tuple[int, int] = (128, 256),
    use_rgb: bool = True,
    rgb_resolution: Tuple[int, int] = (128, 256),
) -> DataLoader:
    """Create a dataloader for the Rice dataset.

    Split rules
    -----------
    train : Rice sequences NOT in test-rice (split.json), with an additional
            case-insensitive exclusion for every Keck or Dell sequence
    val   : Rice sequences listed in test-rice (split.json)
    test  : All sequences discovered under smoke_eval_root_dir

    Args:
        rice_root_dir: Root directory for Rice dataset (required for train/val)
        smoke_eval_root_dir: Root directory for Smoke-Eval test sequences.
            Required when split="test".
        split: "train", "val", or "test"
        batch_size: Batch size
        shuffle: Whether to shuffle (train/val only; test is always sequential)
        num_workers: Number of data loading workers
        frame_skip: Frame sampling stride
        num_frames: Number of frames F in the causal sliding window.
        drop_last: Drop the last incomplete batch (use True for DDP training).
        seed: Shared shuffle seed across all DDP ranks.
        scale_factor: Radar amplitude scale factor
        max_depth_m: Maximum depth in meters
        depth_resolution: (height, width) for depth resizing
        use_rgb: Whether to include RGB
        rgb_resolution: (height, width) for RGB resizing

    Returns:
        DataLoader instance
    """
    split_config = _load_split_config()
    val_rice_seqs = split_config.get("test-rice", [])
    val_rice_seq_keys = {seq.casefold() for seq in val_rice_seqs}

    proc_kwargs = {
        "scale_factor": scale_factor,
        "max_depth_m": max_depth_m,
        "depth_resolution": depth_resolution,
        "use_rgb": use_rgb,
        "rgb_resolution": rgb_resolution,
    }

    if split in ("train", "val"):
        if rice_root_dir is None:
            raise ValueError("rice_root_dir is required for split='train' or 'val'")
        rice_path = Path(rice_root_dir)
        if not rice_path.exists():
            raise ValueError(f"rice_root_dir does not exist: {rice_path}")

        all_rice_seqs = sorted(
            d.name
            for d in rice_path.iterdir()
            if d.is_dir()
            and (d / "radar.npy").exists()
            and (d / "zed_depth.npy").exists()
        )

        if split == "val":
            sequences = [s for s in all_rice_seqs if s.casefold() in val_rice_seq_keys]
        else:
            sequences = [
                s
                for s in all_rice_seqs
                if s.casefold() not in val_rice_seq_keys
                and _HELD_OUT_BUILDING_PATTERN.search(s) is None
            ]

        if not sequences:
            raise ValueError(f"No valid Rice sequences found for split '{split}'")

        dataset = RiceDataset(
            str(rice_path),
            sequences=sequences,
            frame_skip=frame_skip,
            num_frames=num_frames,
            **proc_kwargs,
        )

    else:  # test
        if smoke_eval_root_dir is None:
            raise ValueError("smoke_eval_root_dir is required for split='test'")
        smoke_path = Path(smoke_eval_root_dir)
        if not smoke_path.exists():
            raise ValueError(f"smoke_eval_root_dir does not exist: {smoke_path}")

        smoke_seqs = sorted(
            d.name
            for d in smoke_path.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
        if not smoke_seqs:
            raise ValueError("No sequences found in smoke_eval_root_dir")

        dataset = RiceDataset(
            str(smoke_path),
            sequences=smoke_seqs,
            frame_skip=frame_skip,
            num_frames=num_frames,
            **proc_kwargs,
        )

    use_shuffle = shuffle and split in ("train", "val")
    generator = None
    if use_shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=use_shuffle,
        generator=generator,
        num_workers=num_workers,
        drop_last=drop_last,
    )


# ── Convenience helpers ────────────────────────────────────────────────────────


def create_train_dataloader(**kwargs) -> DataLoader:
    """Training dataloader (Rice non-val sequences)."""
    return create_dataloader(split="train", **kwargs)


def create_val_dataloader(**kwargs) -> DataLoader:
    """Validation dataloader (Rice test-rice sequences from split.json)."""
    return create_dataloader(split="val", **kwargs)


def create_test_dataloader(**kwargs) -> DataLoader:
    """Test dataloader (all Smoke-Eval sequences)."""
    return create_dataloader(split="test", shuffle=False, **kwargs)


