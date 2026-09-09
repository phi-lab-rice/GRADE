import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collate_fn_helpers import dji_rgb_collator, radar_collator, depth_collator
from torch.utils.data import Dataset


class RiceDataset(Dataset):
    """Dataset for radar, DJI RGB, and ZED depth.

    No-doppler ablation: loads radar_no_doppler.npy (N, 1, elevation, azimuth,
    range) and repeats the single doppler bin 64x so downstream code sees the
    standard (64, elevation, azimuth, range) cube.
    """

    REQUIRED_FILES = ("radar_no_doppler.npy", "dji_rgb.npy", "zed_depth.npy")
    DOPPLER_BINS = 64

    def __init__(
        self,
        root_dir: str,
        sequences: Optional[List[str]] = None,
        frame_skip: int = 1,
        depth_in_meters: bool = True,
        rgb_normalize: bool = True,
        # Processing parameters
        scale_factor: float = 0.001,
        max_depth_m: float = 11.2,
        depth_resolution: Tuple[int, int] = (128, 256),
        use_rgb: bool = True,
        rgb_resolution: Tuple[int, int] = (128, 256),
    ):
        self.root_dir = Path(root_dir)
        self.frame_skip = max(1, frame_skip)
        self.depth_in_meters = depth_in_meters
        self.rgb_normalize = rgb_normalize

        # Processing parameters
        self.proc_params = {
            "scale_factor": scale_factor,
            "max_depth_m": max_depth_m,
            "depth_res": depth_resolution,
            "use_rgb": use_rgb,
            "rgb_res": rgb_resolution,
        }

        self.sequences = self._discover_sequences(sequences)
        self.index_map: List[Tuple[str, int]] = []
        self._seq_arrays: Dict[str, Dict] = {}

        self._build_index()

    def _discover_sequences(self, sequences: Optional[List[str]] = None) -> List[str]:
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Root directory not found: {self.root_dir}")

        all_seqs = sorted(
            d.name
            for d in self.root_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

        # Required files always needed
        required = ["radar_no_doppler.npy", "zed_depth.npy"]
        # Add RGB if use_rgb is True
        if self.proc_params["use_rgb"]:
            required.append("dji_rgb.npy")

        valid = [
            name
            for name in all_seqs
            if all((self.root_dir / name / f).exists() for f in required)
        ]
        if sequences is not None:
            valid = [s for s in valid if s in sequences]
        return valid

    def _build_index(self) -> None:
        self.index_map.clear()
        for seq_name in self.sequences:
            radar = np.load(
                self.root_dir / seq_name / "radar_no_doppler.npy", mmap_mode="r"
            )
            for i in range(0, radar.shape[0], self.frame_skip):
                self.index_map.append((seq_name, i))

    def _load_sequence_arrays(self, seq_name: str) -> Dict:
        if seq_name not in self._seq_arrays:
            seq_dir = self.root_dir / seq_name
            arrays = {
                "radar": np.load(seq_dir / "radar_no_doppler.npy", mmap_mode="r"),
                "depth": np.load(seq_dir / "zed_depth.npy", mmap_mode="r"),
            }
            # Only load RGB if needed
            if self.proc_params["use_rgb"]:
                arrays["rgb"] = np.load(seq_dir / "dji_rgb.npy", mmap_mode="r")
            self._seq_arrays[seq_name] = arrays
        return self._seq_arrays[seq_name]

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq_name, frame_idx = self.index_map[idx]
        arrs = self._load_sequence_arrays(seq_name)

        # === Radar (always needed) ===
        # (1, elevation, azimuth, range) -> repeat single doppler bin to
        # (64, elevation, azimuth, range) so downstream code is unchanged
        radar = np.asarray(arrs["radar"][frame_idx].copy())
        radar = np.repeat(radar, self.DOPPLER_BINS, axis=0)
        radar_amp = torch.from_numpy(np.abs(radar).astype(np.float32))
        radar_phase = torch.from_numpy((np.angle(radar) / np.pi).astype(np.float32))

        processed_radar = radar_collator(
            radar_amp.unsqueeze(0),
            radar_phase.unsqueeze(0),
            scale_factor=self.proc_params["scale_factor"],
        ).squeeze(0)

        # === Depth (always needed) ===
        depth = np.asarray(arrs["depth"][frame_idx]).astype(np.float32)
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

        # === RGB (only if use_rgb is True) ===
        out = {
            "radar": processed_radar,
            "depth": processed_depth,
            "sequence": seq_name,
            "frame_idx": frame_idx,
        }

        if self.proc_params["use_rgb"]:
            rgb = np.asarray(arrs["rgb"][frame_idx])
            rgb = np.transpose(rgb, (2, 0, 1))
            if self.rgb_normalize:
                rgb = rgb.astype(np.float32) / 255.0
            rgb_tensor = torch.from_numpy(rgb)

            out["rgb"] = dji_rgb_collator(
                rgb_tensor.unsqueeze(0),
                target_size=self.proc_params["rgb_res"],
            ).squeeze(0)

        return out


