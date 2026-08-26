import json
from typing import Optional, List, Dict, Tuple, Literal
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset
import torch

from iq1m_dataset import IQ1MMultiModalDataset
from rice_dataset import RiceDataset


def _load_split_config() -> Dict[str, List[str]]:
    """Load split configuration from split.json."""
    split_file = Path(__file__).parent / "split.json"
    if not split_file.exists():
        raise FileNotFoundError(f"Split configuration not found: {split_file}")

    with open(split_file, "r") as f:
        split_config = json.load(f)

    return split_config


def _get_train_sequences(
    all_sequences: List[str], val_sequences: List[str]
) -> List[str]:
    """Deterministically assign all non-validation sequences to training.

    Args:
        all_sequences: All available sequences
        val_sequences: Sequences reserved for validation (from split.json)

    Returns:
        List of training sequences
    """
    val_set = set(val_sequences)
    return [s for s in all_sequences if s not in val_set]


def _discover_rice_sequences(root: Path) -> List[str]:
    """Discover valid sequences under a directory."""
    return sorted(
        d.name
        for d in root.iterdir()
        if d.is_dir()
        and (d / "radar_no_doppler.npy").exists()
        and (d / "dji_rgb.npy").exists()
        and (d / "zed_depth.npy").exists()
    )


def create_dataloader(
    dataset_type: Literal["iq1m", "rice", "concat"],
    iq1m_root_dir: Optional[str] = None,
    rice_root_dir: Optional[str] = None,
    split: Literal["train", "val", "test"] = "train",
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    frame_skip: int = 1,
    test_root: Optional[str] = None,
    # Processing parameters (passed to both datasets)
    scale_factor: float = 0.001,
    max_depth_m: float = 11.2,
    depth_resolution: Tuple[int, int] = (128, 256),
    use_rgb: bool = True,
    rgb_resolution: Tuple[int, int] = (128, 256),
) -> DataLoader:
    """Create a dataloader for IQ1M, Rice, or concatenated datasets.

    Split semantics:
    - Validation: the sequences listed in split.json (keys "val-iq1m"/"val-rice",
      or legacy "test-iq1m"/"test-rice" — historically named "test" but used
      for validation).
    - Train: all valid discovered sequences not in the validation lists
      (deterministic; no random carve-out).
    - Test: a separate dataset directory (test_root); all valid sequences
      found there are used. dataset_type is
      ignored for the test split.

    Args:
        dataset_type: Type of dataset - "iq1m", "rice", or "concat" (train/val only)
        iq1m_root_dir: Root directory for IQ1M dataset (required for "iq1m" or "concat")
        rice_root_dir: Root directory for Rice dataset (required for "rice" or "concat")
        split: Which split to use - "train", "val", or "test"
        batch_size: Batch size
        shuffle: Whether to shuffle (only used for train/val, test uses sequential)
        num_workers: Number of data loading workers
        frame_skip: Frame sampling stride
        test_root: Separate held-out test dataset directory; required for
            split="test"
        scale_factor: Radar amplitude scale factor
        max_depth_m: Maximum depth in meters
        depth_resolution: (width, height) for depth resizing
        use_rgb: Whether to include RGB
        rgb_resolution: (height, width) for RGB resizing

    Returns:
        DataLoader instance
    """
    # Common processing parameters
    proc_kwargs = {
        "scale_factor": scale_factor,
        "max_depth_m": max_depth_m,
        "depth_resolution": depth_resolution,
        "use_rgb": use_rgb,
        "rgb_resolution": rgb_resolution,
    }

    # Test split: separate held-out dataset directory.
    if split == "test":
        if test_root is None or not Path(test_root).exists():
            raise ValueError(
                f"No test dataset: test_root not set or does not exist ({test_root!r}). "
                "The test set is a separate dataset directory."
            )
        test_path = Path(test_root)
        test_seqs = _discover_rice_sequences(test_path)
        if not test_seqs:
            raise ValueError(f"No valid test sequences found under {test_root}")
        test_dataset = RiceDataset(
            str(test_path),
            sequences=test_seqs,
            frame_skip=frame_skip,
            **proc_kwargs,
        )
        return DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    # Load split configuration for train/val
    split_config = _load_split_config()
    # "val-*" preferred; legacy split.json files use "test-*" for the val lists
    val_rice = split_config.get("val-rice", split_config.get("test-rice", []))
    val_iq1m = split_config.get("val-iq1m", split_config.get("test-iq1m", []))

    # Determine sequences based on split
    datasets = []

    if dataset_type == "iq1m":
        if iq1m_root_dir is None:
            raise ValueError("iq1m_root_dir is required for dataset_type='iq1m'")
        path = Path(iq1m_root_dir)
    elif dataset_type == "rice":
        if rice_root_dir is None:
            raise ValueError("rice_root_dir is required for dataset_type='rice'")
        path = Path(rice_root_dir)
    else:  # concat
        path = None

    if dataset_type in ["iq1m", "concat"]:
        iq1m_path = path if dataset_type == "iq1m" else Path(iq1m_root_dir) if iq1m_root_dir else None
        if iq1m_path and iq1m_path.exists():
            # Discover all available IQ1M sequences
            depth_dir = iq1m_path / "metric_depth"
            radar_dir = iq1m_path / "radar_no_doppler"
            video_dir = iq1m_path / "video"

            if depth_dir.exists() and radar_dir.exists() and video_dir.exists():
                depth_seqs = set(d.name for d in depth_dir.iterdir() if d.is_dir())
                radar_seqs = set(d.name for d in radar_dir.iterdir() if d.is_dir())
                video_seqs = set(d.name for d in video_dir.iterdir() if d.is_dir())
                all_iq1m_seqs = sorted(depth_seqs & radar_seqs & video_seqs)

                if split == "val":
                    sequences = [s for s in val_iq1m if s in all_iq1m_seqs]
                else:  # train
                    sequences = _get_train_sequences(all_iq1m_seqs, val_iq1m)

                if sequences:
                    datasets.append(
                        IQ1MMultiModalDataset(
                            str(iq1m_path),
                            sequences=sequences,
                            frame_skip=frame_skip,
                            **proc_kwargs,
                        )
                    )

    if dataset_type in ["rice", "concat"]:
        rice_path = path if dataset_type == "rice" else Path(rice_root_dir) if rice_root_dir else None
        if rice_path and rice_path.exists():
            # Discover all available Rice sequences
            all_rice_seqs = _discover_rice_sequences(rice_path)

            if split == "val":
                sequences = [s for s in val_rice if s in all_rice_seqs]
            else:  # train
                sequences = _get_train_sequences(all_rice_seqs, val_rice)

            if sequences:
                datasets.append(
                    RiceDataset(
                        str(rice_path),
                        sequences=sequences,
                        frame_skip=frame_skip,
                        **proc_kwargs,
                    )
                )

    if len(datasets) == 0:
        raise ValueError(
            f"No valid datasets found for split '{split}'"
        )

    combined_dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    # Test split uses sequential sampling, train/val uses shuffled
    use_shuffle = shuffle and split in ["train", "val"]

    return DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=use_shuffle,
        num_workers=num_workers,
    )


# Convenience functions for IQ1M dataset
def create_iq1m_train_dataloader(**kwargs) -> DataLoader:
    """Create IQ1M training dataloader."""
    return create_dataloader(dataset_type="iq1m", split="train", **kwargs)


def create_iq1m_val_dataloader(**kwargs) -> DataLoader:
    """Create IQ1M validation dataloader."""
    return create_dataloader(dataset_type="iq1m", split="val", **kwargs)


def create_iq1m_test_dataloader(**kwargs) -> DataLoader:
    """Create IQ1M test dataloader."""
    return create_dataloader(dataset_type="iq1m", split="test", shuffle=False, **kwargs)


# Convenience functions for Rice dataset
def create_rice_train_dataloader(**kwargs) -> DataLoader:
    """Create Rice training dataloader."""
    return create_dataloader(dataset_type="rice", split="train", **kwargs)


def create_rice_val_dataloader(**kwargs) -> DataLoader:
    """Create Rice validation dataloader."""
    return create_dataloader(dataset_type="rice", split="val", **kwargs)


def create_rice_test_dataloader(**kwargs) -> DataLoader:
    """Create Rice test dataloader."""
    return create_dataloader(dataset_type="rice", split="test", shuffle=False, **kwargs)


# Convenience functions for concatenated dataset (IQ1M + Rice)
def create_concat_train_dataloader(**kwargs) -> DataLoader:
    """Create concatenated IQ1M + Rice training dataloader."""
    return create_dataloader(dataset_type="concat", split="train", **kwargs)


def create_concat_val_dataloader(**kwargs) -> DataLoader:
    """Create concatenated IQ1M + Rice validation dataloader."""
    return create_dataloader(dataset_type="concat", split="val", **kwargs)


def create_concat_test_dataloader(**kwargs) -> DataLoader:
    """Create test dataloader from the separate test_root directory."""
    return create_dataloader(
        dataset_type="concat", split="test", shuffle=False, **kwargs
    )


# Aliases for backwards compatibility
def create_iq1m_dataloader(**kwargs) -> DataLoader:
    """Create IQ1M dataloader (defaults to train split)."""
    return create_dataloader(dataset_type="iq1m", split="train", **kwargs)


def create_rice_dataloader(**kwargs) -> DataLoader:
    """Create Rice dataloader (defaults to train split)."""
    return create_dataloader(dataset_type="rice", split="train", **kwargs)


def create_concat_dataloader(**kwargs) -> DataLoader:
    """Create concatenated IQ1M + Rice dataloader (defaults to train split)."""
    return create_dataloader(dataset_type="concat", split="train", **kwargs)


