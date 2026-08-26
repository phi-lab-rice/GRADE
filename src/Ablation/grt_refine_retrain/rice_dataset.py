import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collate_fn_helpers import dji_rgb_collator, radar_collator, depth_collator
from torch.utils.data import Dataset


class RiceDataset(Dataset):
    """Dataset for radar, DJI RGB, and ZED depth.

    Supports causal sliding-window data loading. For target frame at index i,
    returns frames [i-(F-1), ..., i-1, i]. When i < F-1, the window is padded
    by repeating the first available frame of the sequence.

    Each tensor always has a leading frame dimension (F, ...), even when F=1.
    """

    # These files are now strictly required.
    REQUIRED_FILES_BASE = ("radar.npy", "zed_depth.npy")
    RGB_CANDIDATES = ("dji_rgb.npy", "zed_rgb.npy")

    def __init__(
        self,
        root_dir: str,
        sequences: Optional[List[str]] = None,
        frame_skip: int = 1,
        depth_in_meters: bool = True,
        rgb_normalize: bool = True,
        frame_split: Optional[str] = None,  # 'train' | 'val' | None for all frames
        frame_val_ratio: float = 0.2,  # fraction of frames reserved for val
        # Processing parameters
        scale_factor: float = 0.001,
        max_depth_m: float = 11.2,
        depth_resolution: Tuple[int, int] = (128, 256),
        use_rgb: bool = True,
        rgb_resolution: Tuple[int, int] = (128, 256),
        dji_calibrate: bool = True,
        num_frames: int = 1,
    ):
        self.root_dir = Path(root_dir)
        self.frame_skip = max(1, frame_skip)
        self.depth_in_meters = depth_in_meters
        self.rgb_normalize = rgb_normalize
        self.frame_split = frame_split
        self.frame_val_ratio = frame_val_ratio
        self.num_frames = max(1, num_frames)

        # Processing parameters
        self.proc_params = {
            "scale_factor": scale_factor,
            "max_depth_m": max_depth_m,
            "depth_res": depth_resolution,
            "use_rgb": use_rgb,
            "rgb_res": rgb_resolution,
            "dji_calibrate": dji_calibrate,
        }

        self.sequences = self._discover_sequences(sequences)
        self.index_map: List[Tuple[str, int]] = []
        self._seq_arrays: Dict[str, Dict] = {}

        self._build_index()

        # Frame-level split: slice the flat index_map deterministically.
        # Sequences are sorted, so the order is identical on every DDP rank.
        if self.frame_split is not None:
            n_total = len(self.index_map)
            n_val = int(n_total * self.frame_val_ratio)
            n_train = n_total - n_val
            if self.frame_split == "train":
                self.index_map = self.index_map[:n_train]
            elif self.frame_split == "val":
                self.index_map = self.index_map[n_train:]

    def _discover_sequences(self, sequences: Optional[List[str]] = None) -> List[str]:
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Root directory not found: {self.root_dir}")

        all_seqs = sorted(
            d.name
            for d in self.root_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

        # Required files: both radar and depth are now mandatory.
        required_base = ["zed_depth.npy", "radar.npy"]

        def _has_rgb(seq_name: str) -> bool:
            return any(
                (self.root_dir / seq_name / f).exists() for f in self.RGB_CANDIDATES
            )

        valid = [
            name
            for name in all_seqs
            if all((self.root_dir / name / f).exists() for f in required_base)
            and (not self.proc_params["use_rgb"] or _has_rgb(name))
        ]
        if sequences is not None:
            valid = [s for s in valid if s in sequences]
        return valid

    def _build_index(self) -> None:
        self.index_map.clear()
        for seq_name in self.sequences:
            seq_dir = self.root_dir / seq_name
            radar = np.load(seq_dir / "radar.npy", mmap_mode="r")
            depth = np.load(seq_dir / "zed_depth.npy", mmap_mode="r")
            n_frames = min(radar.shape[0], depth.shape[0])
            if self.proc_params["use_rgb"]:
                for rgb_name in self.RGB_CANDIDATES:
                    rgb_path = seq_dir / rgb_name
                    if rgb_path.exists():
                        rgb = np.load(rgb_path, mmap_mode="r")
                        n_frames = min(n_frames, rgb.shape[0])
                        break
            for i in range(0, n_frames, self.frame_skip):
                self.index_map.append((seq_name, i))

    def _load_sequence_arrays(self, seq_name: str) -> Dict:
        if seq_name not in self._seq_arrays:
            seq_dir = self.root_dir / seq_name
            arrays = {
                "depth": np.load(seq_dir / "zed_depth.npy", mmap_mode="r"),
                "radar": np.load(seq_dir / "radar.npy", mmap_mode="r"),
            }
            # Load RGB: prefer dji_rgb.npy, fall back to zed_rgb.npy
            if self.proc_params["use_rgb"]:
                for rgb_name in self.RGB_CANDIDATES:
                    rgb_path = seq_dir / rgb_name
                    if rgb_path.exists():
                        arrays["rgb"] = np.load(rgb_path, mmap_mode="r")
                        break
                else:
                    raise FileNotFoundError(
                        f"No RGB file found in {seq_dir} "
                        f"(tried: {self.RGB_CANDIDATES})"
                    )
            self._seq_arrays[seq_name] = arrays
        return self._seq_arrays[seq_name]

    def __len__(self) -> int:
        return len(self.index_map)

    def _get_window_indices(self, frame_idx: int) -> List[int]:
        """Return causal window indices [i-(F-1), ..., i] with left-padding.

        Frames before the start of the sequence are clamped to frame 0.
        """
        indices = []
        for offset in range(self.num_frames - 1, -1, -1):
            idx = frame_idx - offset
            indices.append(max(0, idx))
        return indices

    def _process_single_frame(self, arrs: Dict, fi: int) -> Dict[str, torch.Tensor]:
        """Process a single frame index and return processed tensors."""
        # === Radar ===
        radar = np.asarray(arrs["radar"][fi].copy())
        radar_amp = torch.from_numpy(np.abs(radar).astype(np.float32))
        radar_phase = torch.from_numpy(
            (np.angle(radar) / np.pi).astype(np.float32)
        )

        processed_radar = radar_collator(
            radar_amp.unsqueeze(0),
            radar_phase.unsqueeze(0),
            scale_factor=self.proc_params["scale_factor"],
        ).squeeze(0)

        # === Depth ===
        depth = np.asarray(arrs["depth"][fi]).astype(np.float32)
        if self.depth_in_meters:
            depth = depth / 1000.0
        invalid = ~(np.isfinite(depth) & (depth > 0))
        depth[invalid] = 0.0
        depth = depth[np.newaxis, ...]
        depth_tensor = torch.from_numpy(depth).float()

        processed_depth = depth_collator(
            depth_tensor.unsqueeze(0),
            max_depth_m=self.proc_params["max_depth_m"],
            target_size=self.proc_params["depth_res"],
        ).squeeze(0)

        frame_out: Dict[str, torch.Tensor] = {
            "depth": processed_depth,
        }
        if processed_radar is not None:
            frame_out["radar"] = processed_radar

        # === RGB ===
        if self.proc_params["use_rgb"]:
            rgb = np.asarray(arrs["rgb"][fi])
            rgb = np.transpose(rgb, (2, 0, 1))
            if self.rgb_normalize:
                rgb = rgb.astype(np.float32) / 255.0
            rgb_tensor = torch.from_numpy(rgb)

            frame_out["rgb"] = dji_rgb_collator(
                rgb_tensor.unsqueeze(0),
                target_size=self.proc_params["rgb_res"],
                calibrate=False,
            ).squeeze(0)

        return frame_out

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq_name, frame_idx = self.index_map[idx]
        arrs = self._load_sequence_arrays(seq_name)

        # Always return tensors with a leading frame dimension (F, ...)
        window_indices = self._get_window_indices(frame_idx)
        frame_outs = [self._process_single_frame(arrs, fi) for fi in window_indices]

        # Stack each key along a new leading frame dimension
        keys = frame_outs[0].keys()
        out = {k: torch.stack([f[k] for f in frame_outs], dim=0) for k in keys}
        out["sequence"] = seq_name
        out["frame_idx"] = frame_idx
        return out


