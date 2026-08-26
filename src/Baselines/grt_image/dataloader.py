"""
Dataloader for MobiCom processed dataset (output of processor.py).

Uses the optimized format produced by processor.py:
- radar.npy: (N, doppler, elevation, azimuth, range) complex64
- dji_rgb.npy: (N, H, W, 3) uint8
- zed_depth.npy: (N, H, W) uint16, depth in millimeters

This module provides:
- `RiceDataset`: frame-level dataset returning radar amplitude/phase, DJI RGB,
  and ZED depth (ground truth).
- `create_rice_dataloader`: generic dataloader for an arbitrary set of sequences.
- `create_train_val_test_loaders`: uses the configured split file for fixed
  validation sequences and a separate Smoke-Eval root for testing.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class RiceDataset(Dataset):
    """
    Dataset for processor.py output: radar, DJI RGB, and ZED depth per frame.

    Args:
        root_dir: Root directory containing sequence subdirs (e.g. processed/),
            each with radar.npy, dji_rgb.npy, zed_depth.npy.
        sequences: Optional list of sequence names to load. If None, loads all
            subdirs that contain the three required files.
        frame_skip: Sample every frame_skip frames (1 = all frames).
        return_radar_complex: If True, return radar as complex tensor; if False,
            return radar_amplitude and radar_phase as separate float tensors.
        depth_in_meters: If True, convert depth from mm to meters.
        rgb_normalize: If True, return RGB in [0, 1] float; else uint8 [0, 255].
    """

    REQUIRED_FILES = ("radar.npy", "dji_rgb.npy", "zed_depth.npy")

    def __init__(
        self,
        root_dir: str,
        sequences: Optional[List[str]] = None,
        frame_skip: int = 1,
        return_radar_complex: bool = False,
        depth_in_meters: bool = True,
        rgb_normalize: bool = True,
        image_height: int = 288,
        image_width: int = 512,
    ):
        self.root_dir = Path(root_dir)
        self.frame_skip = max(1, frame_skip)
        self.return_radar_complex = return_radar_complex
        self.depth_in_meters = depth_in_meters
        self.rgb_normalize = rgb_normalize
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        if self.image_height <= 0 or self.image_width <= 0:
            raise ValueError("image_height and image_width must be positive")

        self.sequences = self._discover_sequences(sequences)
        self.index_map: List[Tuple[str, int]] = []  # (seq_name, frame_idx)
        self._seq_arrays: Dict[str, Dict] = {}  # seq -> {radar, depth, dji_rgb}

        self._build_index()

    def _discover_sequences(self, sequences: Optional[List[str]] = None) -> List[str]:
        """Return list of sequence names that have all required files."""
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Root directory not found: {self.root_dir}")

        all_seqs = sorted(
            d.name
            for d in self.root_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
        valid = []
        for name in all_seqs:
            seq_dir = self.root_dir / name
            if all((seq_dir / f).exists() for f in self.REQUIRED_FILES):
                valid.append(name)
        if sequences is not None:
            valid = [s for s in valid if s in sequences]
        return valid

    def _build_index(self) -> None:
        """Build (seq_name, frame_idx) index, using radar.npy for frame count."""
        self.index_map.clear()
        for seq_name in self.sequences:
            seq_dir = self.root_dir / seq_name
            radar_path = seq_dir / "radar.npy"
            arrays = self._load_sequence_arrays(seq_name)
            n_frames = min(array.shape[0] for array in arrays.values())
            for i in range(0, n_frames, self.frame_skip):
                self.index_map.append((seq_name, i))

    def _load_sequence_arrays(self, seq_name: str) -> Dict:
        """Lazy-load or return cached arrays for a sequence."""
        if seq_name not in self._seq_arrays:
            seq_dir = self.root_dir / seq_name
            self._seq_arrays[seq_name] = {
                "radar": np.load(seq_dir / "radar.npy", mmap_mode="r"),
                "rgb": np.load(seq_dir / "dji_rgb.npy", mmap_mode="r"),
                "depth": np.load(seq_dir / "zed_depth.npy", mmap_mode="r"),
            }
        return self._seq_arrays[seq_name]

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq_name, frame_idx = self.index_map[idx]
        arrs = self._load_sequence_arrays(seq_name)

        rgb = np.asarray(arrs["rgb"][frame_idx]).copy()
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f"Expected RGB frame shaped [H, W, 3], got {rgb.shape}")
        # (H, W) uint16 mm (processor saves as uint16)
        depth = np.asarray(arrs["depth"][frame_idx]).astype(np.float32)
        # (doppler, elevation, azimuth, range) complex64
        radar = np.asarray(arrs["radar"][frame_idx]).copy()

        # Depth: uint16 mm -> float; optional mm -> m; handle invalid
        if self.depth_in_meters:
            depth = depth / 1000.0
        invalid = ~(np.isfinite(depth) & (depth > 0))
        depth[invalid] = 0.0
        depth = depth[np.newaxis, ...]  # (1, H, W)

        # RGB: [H, W, 3] uint8 -> resized [3, image_height, image_width] float.
        image = torch.from_numpy(np.transpose(rgb, (2, 0, 1)).copy()).float()
        if self.rgb_normalize:
            image = image / 255.0
        image = F.interpolate(
            image.unsqueeze(0),
            size=(self.image_height, self.image_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # Radar: amplitude and phase
        radar_amplitude = np.abs(radar).astype(np.float32)
        radar_phase = np.angle(radar).astype(np.float32) / np.pi
        out = {
            "radar_amplitude": torch.from_numpy(radar_amplitude),
            "radar_phase": torch.from_numpy(radar_phase),
            "image": image,
            "depth": torch.from_numpy(depth),
            "sequence": seq_name,
            "frame_idx": frame_idx,
        }
        if self.return_radar_complex:
            out["radar_cube"] = torch.from_numpy(radar.copy())
        # Depth in mm for optional use (1, H, W) float32
        depth_mm = np.asarray(arrs["depth"][frame_idx]).astype(np.float32)
        out["depth_mm"] = torch.from_numpy(depth_mm[np.newaxis, ...])
        return out


def create_rice_dataloader(
    root_dir: str,
    batch_size: int = 8,
    num_workers: int = 0,
    frame_skip: int = 1,
    sequences: Optional[List[str]] = None,
    return_radar_complex: bool = False,
    depth_in_meters: bool = True,
    rgb_normalize: bool = True,
    image_height: int = 288,
    image_width: int = 512,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader for the Rice (processor output) dataset."""
    dataset = RiceDataset(
        root_dir=root_dir,
        sequences=sequences,
        frame_skip=frame_skip,
        return_radar_complex=return_radar_complex,
        depth_in_meters=depth_in_meters,
        rgb_normalize=rgb_normalize,
        image_height=image_height,
        image_width=image_width,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def create_train_val_test_loaders(
    train_root: str,
    split_json_path: Optional[str],
    test_root: str,
    batch_size: int = 8,
    num_workers: int = 0,
    frame_skip: int = 1,
    return_radar_complex: bool = False,
    depth_in_meters: bool = True,
    rgb_normalize: bool = True,
    image_height: int = 288,
    image_width: int = 512,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create fixed training/validation and Smoke-Eval test loaders.

    The ``test`` list in the configured split file is treated as a fixed
    validation sequence list. All other valid training sequences are used
    for training. ``test_root`` is a separately structured Smoke-Eval tree;
    every valid sequence it contains is evaluated only as the test set.
    """
    if split_json_path is None:
        split_path = Path(__file__).resolve().parent / "split.json"
    else:
        split_path = Path(split_json_path)
        if not split_path.exists() and not split_path.is_absolute():
            fallback = Path(__file__).resolve().parent / split_path.name
            if fallback.exists():
                split_path = fallback

    with split_path.open("r") as f:
        split = json.load(f)
    validation_sequences = split.get("test", [])

    discovered_train = RiceDataset(
        root_dir=train_root,
        frame_skip=frame_skip,
        return_radar_complex=return_radar_complex,
        depth_in_meters=depth_in_meters,
        rgb_normalize=rgb_normalize,
        image_height=image_height,
        image_width=image_width,
    )
    validation_set = set(validation_sequences)
    train_sequences = [
        sequence
        for sequence in discovered_train.sequences
        if sequence not in validation_set
    ]
    resolved_validation_sequences = [
        sequence
        for sequence in validation_sequences
        if sequence in discovered_train.sequences
    ]

    dataset_kwargs = {
        "frame_skip": frame_skip,
        "return_radar_complex": return_radar_complex,
        "depth_in_meters": depth_in_meters,
        "rgb_normalize": rgb_normalize,
        "image_height": image_height,
        "image_width": image_width,
    }
    train_dataset = RiceDataset(
        root_dir=train_root, sequences=train_sequences, **dataset_kwargs
    )
    val_dataset = RiceDataset(
        root_dir=train_root,
        sequences=resolved_validation_sequences,
        **dataset_kwargs,
    )
    test_dataset = RiceDataset(root_dir=test_root, sequences=None, **dataset_kwargs)

    loader_kwargs = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": True}
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def create_train_val_loaders(
    train_root: str,
    split_json_path: Optional[str],
    batch_size: int = 8,
    num_workers: int = 0,
    frame_skip: int = 1,
    return_radar_complex: bool = False,
    depth_in_meters: bool = True,
    rgb_normalize: bool = True,
    image_height: int = 288,
    image_width: int = 512,
) -> Tuple[DataLoader, DataLoader]:
    """Create training and fixed validation loaders only."""
    if split_json_path is None:
        split_path = Path(__file__).resolve().parent / "split.json"
    else:
        split_path = Path(split_json_path)
        if not split_path.exists() and not split_path.is_absolute():
            fallback = Path(__file__).resolve().parent / split_path.name
            if fallback.exists():
                split_path = fallback

    with split_path.open("r") as f:
        split = json.load(f)
    validation_sequences = split.get("test", [])

    discovered = RiceDataset(
        root_dir=train_root,
        frame_skip=frame_skip,
        return_radar_complex=return_radar_complex,
        depth_in_meters=depth_in_meters,
        rgb_normalize=rgb_normalize,
        image_height=image_height,
        image_width=image_width,
    )
    validation_set = set(validation_sequences)
    train_sequences = [
        sequence for sequence in discovered.sequences if sequence not in validation_set
    ]
    resolved_validation_sequences = [
        sequence for sequence in validation_sequences if sequence in discovered.sequences
    ]

    dataset_kwargs = {
        "frame_skip": frame_skip,
        "return_radar_complex": return_radar_complex,
        "depth_in_meters": depth_in_meters,
        "rgb_normalize": rgb_normalize,
        "image_height": image_height,
        "image_width": image_width,
    }
    train_dataset = RiceDataset(
        root_dir=train_root, sequences=train_sequences, **dataset_kwargs
    )
    val_dataset = RiceDataset(
        root_dir=train_root,
        sequences=resolved_validation_sequences,
        **dataset_kwargs,
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    return (
        DataLoader(train_dataset, shuffle=True, **loader_kwargs),
        DataLoader(val_dataset, shuffle=False, **loader_kwargs),
    )
