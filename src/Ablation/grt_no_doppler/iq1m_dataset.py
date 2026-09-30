import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from collate_fn_helpers import radar_collator, depth_collator, fisheye_rgb_collator


class IQ1MMultiModalDataset(Dataset):
    """
    Dataset for loading aligned lidar, radar, and video frames.

    No-doppler ablation: radar is read from
    root_dir/radar_no_doppler/<sequence>/amplitude.npy and phase.npy
    (single doppler bin), and the doppler axis is repeated 64x so
    downstream code sees the standard cube.

    Args:
        root_dir: Root directory containing 'lidar', 'radar_no_doppler',
            'video' folders
        sequences: Optional list of sequence names to load. If None, loads all.
        transform: Optional transform to apply to video frames
    """

    DOPPLER_BINS = 64

    def __init__(
        self,
        root_dir: str,
        sequences: Optional[List[str]] = None,
        frame_skip: int = 1,
        split_type: Optional[str] = None,  # 'train', 'val', 'test', or None for all
        # Processing parameters
        scale_factor: float = 0.001,
        max_depth_m: float = 11.2,
        depth_resolution: Tuple[int, int] = (128, 256),
        use_rgb: bool = True,
        rgb_resolution: Tuple[int, int] = (128, 256),
    ):
        self.root_dir = Path(root_dir)
        self.depth_dir = self.root_dir / "metric_depth"
        self.radar_dir = self.root_dir / "radar_no_doppler"
        self.video_dir = self.root_dir / "video"
        self.frame_skip = max(1, frame_skip)
        self.split_type = split_type

        # Processing parameters
        self.proc_params = {
            "scale_factor": scale_factor,
            "max_depth_m": max_depth_m,
            "depth_res": depth_resolution,
            "use_rgb": use_rgb,
            "rgb_res": rgb_resolution,
        }

        # Load split configuration if split_type is specified
        if split_type is not None:
            split_config = self._load_split_config()
            sequences = self._get_sequences_for_split(split_config, sequences)

        # Discover sequences
        self.sequences = self._discover_sequences(sequences)

        # Build index mapping (global_idx -> (sequence_name, frame_idx))
        self.index_map: List[Tuple[str, int]] = []
        self.sequence_info: Dict[str, dict] = {}

        # Memory-mapped numpy arrays for efficient loading
        self._depth_mmap: Dict[str, np.memmap] = {}
        self._radar_amplitude_mmap: Dict[str, np.memmap] = {}
        self._radar_phase_mmap: Dict[str, np.memmap] = {}
        self._video_captures: Dict[str, cv2.VideoCapture] = {}

        self._build_index()

    def _load_split_config(self) -> Dict:
        """Load split configuration from iq1m_split.json"""
        split_file = Path(__file__).parent / "iq1m_split.json"
        if not split_file.exists():
            raise FileNotFoundError(f"Split configuration not found: {split_file}")

        with open(split_file, "r") as f:
            split_config = json.load(f)

        return split_config

    def _get_sequences_for_split(
        self, split_config: Dict, requested_sequences: Optional[List[str]] = None
    ) -> Optional[List[str]]:
        """Get sequences for the specified split type"""
        if self.split_type == "test":
            sequences = split_config.get("test", [])
        elif self.split_type in ["train", "val"]:
            # Get all available sequences
            all_sequences = self._get_all_available_sequences()
            test_sequences = set(split_config.get("test", []))
            # Exclude test sequences
            sequences = [s for s in all_sequences if s not in test_sequences]
        else:
            raise ValueError(
                f"Invalid split_type: {self.split_type}. Must be 'train', 'val', 'test', or None"
            )

        # Filter by requested sequences if provided
        if requested_sequences is not None:
            sequences = [s for s in sequences if s in requested_sequences]

        return sequences

    def _get_all_available_sequences(self) -> List[str]:
        """Get all available sequences from the dataset"""
        depth_seqs = set(
            d.name
            for d in self.depth_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
        radar_seqs = set(
            d.name
            for d in self.radar_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

        # Only consider video if use_rgb is True
        if self.proc_params["use_rgb"]:
            video_seqs = set(
                d.name
                for d in self.video_dir.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            )
            # Find common sequences across all modalities
            common_seqs = depth_seqs & radar_seqs & video_seqs
        else:
            # Only need depth and radar
            common_seqs = depth_seqs & radar_seqs

        return sorted(list(common_seqs))

    def _discover_sequences(self, sequences: Optional[List[str]] = None) -> List[str]:
        """Discover available sequences with required modalities."""
        # Get sequences from each modality folder
        depth_seqs = set(
            d.name
            for d in self.depth_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
        radar_seqs = set(
            d.name
            for d in self.radar_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

        # Only consider video if use_rgb is True
        if self.proc_params["use_rgb"]:
            video_seqs = set(
                d.name
                for d in self.video_dir.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            )
            # Find common sequences across all modalities
            common_seqs = depth_seqs & radar_seqs & video_seqs
        else:
            # Only need depth and radar
            common_seqs = depth_seqs & radar_seqs

        if sequences is not None:
            # Filter to requested sequences
            common_seqs = common_seqs & set(sequences)

        return sorted(list(common_seqs))

    def _build_index(self):
        """Build global index mapping and load metadata."""
        for seq_name in self.sequences:
            # Load metadata from radar_no_doppler (or radar/lidar if available)
            metadata_path = self.radar_dir / seq_name / "metadata.json"
            if not metadata_path.exists():
                metadata_path = self.root_dir / "radar" / seq_name / "metadata.json"
            if not metadata_path.exists():
                metadata_path = self.root_dir / "lidar" / seq_name / "metadata.json"
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            n_frames = metadata["n_frames"]
            self.sequence_info[seq_name] = {
                "n_frames": n_frames,
                "metadata": metadata,
                "start_idx": len(self.index_map),
            }

            # Add frames to index with skipping
            # Range: 0, frame_skip, 2*frame_skip, ...
            for frame_idx in range(0, n_frames, self.frame_skip):
                self.index_map.append((seq_name, frame_idx))

            self.sequence_info[seq_name]["end_idx"] = len(self.index_map)

    def _get_depth_mmap(self, seq_name: str) -> np.memmap:
        """Get or create memory-mapped metric depth array."""
        if seq_name not in self._depth_mmap:
            path = self.depth_dir / seq_name / "metric_depth.npy"
            self._depth_mmap[seq_name] = np.load(path, mmap_mode="r")
        return self._depth_mmap[seq_name]

    def _get_radar_mmap(self, seq_name: str) -> Tuple[np.memmap, np.memmap]:
        """Get or create memory-mapped radar arrays."""
        if seq_name not in self._radar_amplitude_mmap:
            amp_path = self.radar_dir / seq_name / "amplitude.npy"
            phase_path = self.radar_dir / seq_name / "phase.npy"
            self._radar_amplitude_mmap[seq_name] = np.load(amp_path, mmap_mode="r")
            self._radar_phase_mmap[seq_name] = np.load(phase_path, mmap_mode="r")
        return self._radar_amplitude_mmap[seq_name], self._radar_phase_mmap[seq_name]

    def _get_video_capture(self, seq_name: str) -> cv2.VideoCapture:
        """Get or create video capture object."""
        if seq_name not in self._video_captures:
            video_path = self.video_dir / seq_name / "video.avi"
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {video_path}")
            self._video_captures[seq_name] = cap
        return self._video_captures[seq_name]

    def _load_rgb_frame(self, seq_name: str, frame_idx: int) -> np.ndarray:
        """Load a specific frame from video."""
        cap = self._get_video_capture(seq_name)

        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            raise RuntimeError(f"Failed to read frame {frame_idx} from {seq_name}")

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        seq_name, frame_idx = self.index_map[idx]

        # === Radar (always needed) ===
        amp_mmap, phase_mmap = self._get_radar_mmap(seq_name)
        radar_amp = torch.from_numpy(amp_mmap[frame_idx].copy()).float()
        radar_phase = torch.from_numpy(phase_mmap[frame_idx].copy()).float()
        # Single doppler bin -> repeat to the standard 64-bin cube so
        # downstream code is unchanged
        radar_amp = torch.repeat_interleave(radar_amp, self.DOPPLER_BINS, dim=0)
        radar_phase = torch.repeat_interleave(radar_phase, self.DOPPLER_BINS, dim=0)

        processed_radar = radar_collator(
            radar_amp.unsqueeze(0),
            radar_phase.unsqueeze(0),
            scale_factor=self.proc_params["scale_factor"],
        ).squeeze(0)

        # === Depth (always needed) ===
        depth_mmap = self._get_depth_mmap(seq_name)
        depth = torch.from_numpy(depth_mmap[frame_idx].copy()).float().unsqueeze(0)

        processed_depth = depth_collator(
            depth.unsqueeze(0),
            max_depth_m=self.proc_params["max_depth_m"],
            target_size=self.proc_params["depth_res"],
        ).squeeze(0)

        out = {
            "radar": processed_radar,
            "depth": processed_depth,
            "sequence": seq_name,
            "frame_idx": frame_idx,
        }

        # === RGB (only if use_rgb is True) ===
        if self.proc_params["use_rgb"]:
            rgb = torch.from_numpy(
                self._load_rgb_frame(seq_name, frame_idx)
            ).float().permute(2, 0, 1) / 255.0

            out["rgb"] = fisheye_rgb_collator(
                rgb.unsqueeze(0),
                target_size=self.proc_params["rgb_res"],
            ).squeeze(0)

        return out

    def get_sequence_frames(self, seq_name: str) -> List[int]:
        """Get global indices for all frames in a sequence."""
        info = self.sequence_info[seq_name]
        return list(range(info["start_idx"], info["end_idx"]))

    def close(self):
        """Release video capture resources."""
        for cap in self._video_captures.values():
            cap.release()
        self._video_captures.clear()

    def __del__(self):
        self.close()


