import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset


class RiceDataset(Dataset):
    """Raw Rice dataset reader for DJI RGB, ZED depth and radar point clouds.

    This dataset returns raw per-frame arrays and leaves geometric processing to
    `collate_fn_helpers.make_rice_collate_fn`.
    """

    def __init__(
        self,
        base_dir: str,
        split_json_path: Optional[str] = None,
        split: str = "train",
        input_height: int = 288,
        input_width: int = 512,
        patch_size: Optional[Tuple[int, int]] = None,
    ):
        self.base_dir = base_dir
        self.split = split
        self.input_height = int(input_height)
        self.input_width = int(input_width)
        self.patch_size = self._resolve_patch_size(patch_size)

        test_sequences = self._load_test_split(split_json_path)

        all_sequences = sorted(
            d
            for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d)) and not d.startswith(".")
        )

        self.sequences: List[str] = []
        for seq in all_sequences:
            if split == "train" and seq in test_sequences:
                continue
            if split == "test" and seq not in test_sequences:
                continue
            if self._is_valid_sequence(os.path.join(base_dir, seq)):
                self.sequences.append(seq)

        self.dji_rgb_mmaps: Dict[str, np.memmap] = {}
        self.zed_depth_mmaps: Dict[str, np.memmap] = {}
        self.samples: List[Tuple[str, int]] = []

        for seq in self.sequences:
            seq_dir = os.path.join(self.base_dir, seq)
            dji_rgb_path = os.path.join(seq_dir, "dji_rgb.npy")
            zed_depth_path = os.path.join(seq_dir, "zed_depth.npy")

            self.dji_rgb_mmaps[seq] = np.load(dji_rgb_path, mmap_mode="r")
            self.zed_depth_mmaps[seq] = np.load(zed_depth_path, mmap_mode="r")

            n_frames = min(
                len(self.dji_rgb_mmaps[seq]),
                len(self.zed_depth_mmaps[seq]),
            )
            for frame_idx in range(n_frames):
                self.samples.append((seq, frame_idx))

    def _resolve_patch_size(
        self, patch_size: Optional[Tuple[int, int]]
    ) -> Tuple[int, int]:
        if patch_size is not None:
            return int(patch_size[0]), int(patch_size[1])

        # Scale default CaFNet patch size (50, 150) from 352x704.
        base_h, base_w = 352, 704
        scale_h = self.input_height / float(base_h)
        scale_w = self.input_width / float(base_w)
        ext_h = max(1, int(round(50 * scale_h)))
        ext_w = max(1, int(round(150 * scale_w)))
        return ext_h, ext_w

    def _load_test_split(self, split_json_path: Optional[str]) -> set:
        if not split_json_path or not os.path.exists(split_json_path):
            return set()
        with open(split_json_path, "r") as f:
            payload = json.load(f)
        return set(payload.get("test", []))

    def _is_valid_sequence(self, seq_dir: str) -> bool:
        dji_rgb_path = os.path.join(seq_dir, "dji_rgb.npy")
        zed_depth_path = os.path.join(seq_dir, "zed_depth.npy")
        pcd_dir = os.path.join(seq_dir, "pcd")
        return (
            os.path.exists(dji_rgb_path)
            and os.path.exists(zed_depth_path)
            and os.path.isdir(pcd_dir)
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        seq, frame_idx = self.samples[idx]
        seq_dir = os.path.join(self.base_dir, seq)

        dji_rgb = np.asarray(self.dji_rgb_mmaps[seq][frame_idx]).copy()
        zed_depth_mm = np.asarray(self.zed_depth_mmaps[seq][frame_idx]).copy()

        pcd_path = os.path.join(seq_dir, "pcd", f"pcd_{frame_idx}.npy")
        if os.path.exists(pcd_path):
            radar_pcd_xyz = np.asarray(np.load(pcd_path), dtype=np.float32)
        else:
            radar_pcd_xyz = np.zeros((0, 3), dtype=np.float32)

        return {
            "sample_idx": idx,
            "sequence": seq,
            "frame_idx": frame_idx,
            "dji_rgb": dji_rgb,
            "zed_depth_mm": zed_depth_mm,
            "radar_pcd_xyz": radar_pcd_xyz,
        }
