"""
Dataloader for MobiCom processed dataset (output of processor.py).

No-doppler ablation: loads radar_no_doppler.npy (N, 1, elevation, azimuth, range)
and repeats the single doppler bin 64x so downstream code sees the standard cube.

Uses the optimized format produced by processor.py:
- radar_no_doppler.npy: (N, 1, elevation, azimuth, range) complex64
- dji_rgb.avi: DJI RGB video (FFV1), (N, H, W, 3) uint8
- zed_depth.npy: (N, H, W) uint16, depth in millimeters

This module provides:
- `RiceDataset`: frame-level dataset returning radar amplitude/phase, DJI RGB,
  and ZED depth (ground truth).
- `create_rice_dataloader`: generic dataloader for an arbitrary set of sequences.
- `create_split_dataloaders`: reads train/val/test split from split.json
  (default: radar_model/split.json) and returns train/val/test dataloaders.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from iq1m_dataset import IQ1MMultiModalDataset


class RiceDataset(Dataset):
    """
    Dataset for processor.py output: radar, DJI RGB, and ZED depth per frame.

    Args:
        root_dir: Root directory containing sequence subdirs (e.g. processed/),
            each with radar.npy, dji_rgb.avi, zed_depth.npy.
        sequences: Optional list of sequence names to load. If None, loads all
            subdirs that contain the three required files.
        frame_skip: Sample every frame_skip frames (1 = all frames).
        return_radar_complex: If True, return radar as complex tensor; if False,
            return radar_amplitude and radar_phase as separate float tensors.
        depth_in_meters: If True, convert depth from mm to meters.
        rgb_normalize: If True, return RGB in [0, 1] float; else uint8 [0, 255].
    """

    REQUIRED_FILES = ("radar_no_doppler.npy", "dji_rgb.npy", "zed_depth.npy")
    DOPPLER_BINS = 64

    def __init__(
        self,
        root_dir: str,
        sequences: Optional[List[str]] = None,
        frame_skip: int = 1,
        return_radar_complex: bool = False,
        depth_in_meters: bool = True,
        rgb_normalize: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.frame_skip = max(1, frame_skip)
        self.return_radar_complex = return_radar_complex
        self.depth_in_meters = depth_in_meters
        self.rgb_normalize = rgb_normalize

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
        """Build (seq_name, frame_idx) index, using radar_no_doppler.npy for frame count."""
        self.index_map.clear()
        for seq_name in self.sequences:
            seq_dir = self.root_dir / seq_name
            radar_path = seq_dir / "radar_no_doppler.npy"
            radar = np.load(radar_path, mmap_mode="r")
            n_frames = radar.shape[0]
            for i in range(0, n_frames, self.frame_skip):
                self.index_map.append((seq_name, i))

    # def _load_video_rgb(self, path: Path) -> np.ndarray:
    #     """Load RGB AVI (e.g. FFV1) as (N, H, W, 3) uint8 RGB."""
    #     cap = cv2.VideoCapture(str(path))
    #     if not cap.isOpened():
    #         raise RuntimeError(f"Failed to open video: {path}")
    #     frames = []
    #     while True:
    #         ret, frame = cap.read()
    #         if not ret:
    #             break
    #         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         frames.append(rgb)
    #     cap.release()
    #     if not frames:
    #         return np.empty((0, 0, 0, 3), dtype=np.uint8)
    #     return np.stack(frames, axis=0)

    def _load_sequence_arrays(self, seq_name: str) -> Dict:
        """Lazy-load or return cached arrays for a sequence."""
        if seq_name not in self._seq_arrays:
            seq_dir = self.root_dir / seq_name
            # dji_rgb = self._load_video_rgb(seq_dir / "dji_rgb.avi")
            self._seq_arrays[seq_name] = {
                "radar": np.load(seq_dir / "radar_no_doppler.npy", mmap_mode="r"),
                "depth": np.load(seq_dir / "zed_depth.npy", mmap_mode="r"),
                # "dji_rgb": dji_rgb,
            }
        return self._seq_arrays[seq_name]

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq_name, frame_idx = self.index_map[idx]
        arrs = self._load_sequence_arrays(seq_name)

        # (H, W, 3) uint8
        # rgb = np.asarray(arrs["dji_rgb"][frame_idx])
        # (H, W) uint16 mm (processor saves as uint16)
        depth = np.asarray(arrs["depth"][frame_idx]).astype(np.float32)
        # (1, elevation, azimuth, range) complex64 -> repeat single doppler bin
        # to (64, elevation, azimuth, range) so downstream code is unchanged
        radar = np.asarray(arrs["radar"][frame_idx]).copy()
        radar = np.repeat(radar, self.DOPPLER_BINS, axis=0)

        # Depth: uint16 mm -> float; optional mm -> m; handle invalid
        if self.depth_in_meters:
            depth = depth / 1000.0
        invalid = ~(np.isfinite(depth) & (depth > 0))
        depth[invalid] = 0.0
        depth = depth[np.newaxis, ...]  # (1, H, W)

        # RGB: (H, W, 3) -> (3, H, W)
        # rgb = np.transpose(rgb, (2, 0, 1))
        # if self.rgb_normalize:
        #     rgb = rgb.astype(np.float32) / 255.0

        # Radar: amplitude and phase
        radar_amplitude = np.abs(radar).astype(np.float32)
        radar_phase = np.angle(radar).astype(np.float32) / np.pi
        out = {
            "radar_amplitude": torch.from_numpy(radar_amplitude),
            "radar_phase": torch.from_numpy(radar_phase),
            # "rgb": torch.from_numpy(rgb),
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
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def create_split_dataloaders(
    root_dir: str,
    split_json_path: Optional[str] = None,
    batch_size: int = 8,
    num_workers: int = 0,
    frame_skip: int = 1,
    return_radar_complex: bool = False,
    depth_in_meters: bool = True,
    rgb_normalize: bool = True,
    test_root: Optional[str] = None,
    seed: Optional[int] = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders.

    Split semantics:
    - Validation: the sequences listed in split.json (key "val", or legacy key
      "test" — historically named "test" but used for validation).
    - Train: the explicit "train" list from split.json if present, otherwise
      all valid sequences under root_dir not in the validation list.
    - Test: a separate dataset directory (test_root); all valid sequences
      found there are used. If test_root is None,
      the test loader is empty.

    Split JSON format:
        { "val": ["seq_x", ...], "train": ["seq_a", ...] }  // "train" optional
        (legacy: { "test": [...] } is read as the val list)

    Returns:
        train_loader, val_loader, test_loader
    """
    if split_json_path is None:
        split_path = Path(__file__).resolve().parent / "split.json"
    else:
        split_path = Path(split_json_path)
        if not split_path.exists() and not split_path.is_absolute():
            # Resolve relative path from this module's directory (e.g. radar_model/)
            fallback = Path(__file__).resolve().parent / split_path.name
            if fallback.exists():
                split_path = fallback

    with split_path.open("r") as f:
        split = json.load(f)

    # "val" preferred; legacy split.json files use "test" for these sequences
    val_sequences = split.get("val", split.get("test", []))
    train_sequences_json = split.get("train", None)

    # Discover all valid sequences in root_dir
    _discover = RiceDataset(
        root_dir=root_dir,
        sequences=None,
        frame_skip=frame_skip,
        return_radar_complex=return_radar_complex,
        depth_in_meters=depth_in_meters,
        rgb_normalize=rgb_normalize,
    )
    val_set = set(val_sequences)
    if train_sequences_json is not None and len(train_sequences_json) > 0:
        # Use explicit train list (intersect with discovered so only valid seqs are used)
        train_sequences = [s for s in train_sequences_json if s in _discover.sequences]
    else:
        # No "train" key: use all discovered sequences not in val
        train_sequences = [s for s in _discover.sequences if s not in val_set]

    train_dataset = RiceDataset(
        root_dir=root_dir,
        sequences=train_sequences,
        frame_skip=frame_skip,
        return_radar_complex=return_radar_complex,
        depth_in_meters=depth_in_meters,
        rgb_normalize=rgb_normalize,
    )

    val_dataset = RiceDataset(
        root_dir=root_dir,
        sequences=[s for s in val_sequences if s in _discover.sequences],
        frame_skip=frame_skip,
        return_radar_complex=return_radar_complex,
        depth_in_meters=depth_in_meters,
        rgb_normalize=rgb_normalize,
    )

    # Test set lives in its own directory (same format); discover all sequences
    if test_root is not None:
        test_dataset = RiceDataset(
            root_dir=test_root,
            sequences=None,
            frame_skip=frame_skip,
            return_radar_complex=return_radar_complex,
            depth_in_meters=depth_in_meters,
            rgb_normalize=rgb_normalize,
        )
    else:
        test_dataset = RiceDataset(
            root_dir=root_dir,
            sequences=[],
            frame_skip=frame_skip,
            return_radar_complex=return_radar_complex,
            depth_in_meters=depth_in_meters,
            rgb_normalize=rgb_normalize,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def create_iq1m_split_dataloaders(
    root_dir: str,
    split_json_path: Optional[str] = None,
    batch_size: int = 8,
    num_workers: int = 0,
    frame_skip: int = 1,
    val_ratio: float = 0.2,
    seed: Optional[int] = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders for the IQ1M dataset using iq1m_split.json.

    Same split semantics as create_split_dataloaders, but backed by
    IQ1MMultiModalDataset (radar already collated by radar_collator, depth
    normalized to [0, 1]). Depth is produced directly at (64, 128) — the GRT
    occupancy resolution — and RGB is skipped.

    Split JSON format:
        { "test": ["seq_x", ...], "train": ["seq_a", ...] }  // "train" optional

    Returns:
        train_loader, val_loader, test_loader
    """
    if split_json_path is None:
        split_path = Path(__file__).resolve().parent / "iq1m_split.json"
    else:
        split_path = Path(split_json_path)
        if not split_path.exists() and not split_path.is_absolute():
            fallback = Path(__file__).resolve().parent / split_path.name
            if fallback.exists():
                split_path = fallback

    with split_path.open("r") as f:
        split = json.load(f)

    test_sequences = split.get("test", [])
    train_sequences_json = split.get("train", None)

    dataset_kwargs = {
        "frame_skip": frame_skip,
        "use_rgb": False,
        "depth_resolution": (64, 128),
    }

    # Discover all valid sequences in root_dir
    _discover = IQ1MMultiModalDataset(root_dir, sequences=None, **dataset_kwargs)
    test_set = set(test_sequences)
    if train_sequences_json is not None and len(train_sequences_json) > 0:
        train_sequences = [s for s in train_sequences_json if s in _discover.sequences]
    else:
        train_sequences = [s for s in _discover.sequences if s not in test_set]

    full_train_dataset = IQ1MMultiModalDataset(
        root_dir, sequences=train_sequences, **dataset_kwargs
    )

    # Random split of training data for validation
    n_total = len(full_train_dataset)
    n_val = int(n_total * val_ratio)
    if n_val == 0 and n_total > 0:
        n_val = 1
    n_train = n_total - n_val

    if n_total == 0:
        train_dataset = full_train_dataset
        val_dataset = IQ1MMultiModalDataset(root_dir, sequences=[], **dataset_kwargs)
    elif seed is None:
        train_dataset, val_dataset = random_split(full_train_dataset, [n_train, n_val])
    else:
        generator = torch.Generator()
        generator.manual_seed(seed)
        train_dataset, val_dataset = random_split(
            full_train_dataset, [n_train, n_val], generator=generator
        )

    test_dataset = IQ1MMultiModalDataset(
        root_dir, sequences=test_sequences, **dataset_kwargs
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
