"""Depth Anything 3 metric-depth inference for Smoke-Eval sequences.

This inference-only adapter follows the official ByteDance-Seed
Depth-Anything-3 Python API. The upstream package supplies the model
architecture; this file supplies the artifact's local weights, camera
calibration, sequence sharding, and output contract.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from accelerate import Accelerator
from safetensors.torch import load_file
from tqdm.auto import tqdm


INTRINSICS = np.array(
    [[365.13, 0.0, 445.43], [0.0, 365.13, 261.18], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)
SCALE_FACTOR = 1.15 * 365.13 / 300.0
TARGET_SIZE = (896, 504)


class Calibrator:
    """Defish DJI frames and map them to the ZED-aligned view."""

    def __init__(self):
        k_dji = np.array(
            [
                [718.48555551, 0.0, 963.36465011],
                [0.0, 720.25844189, 537.87569913],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        d_dji = np.array(
            [0.19022699, 0.03466753, 0.05858962, -0.07070669],
            dtype=np.float64,
        )
        new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            k_dji,
            d_dji,
            (1920, 1080),
            np.eye(3),
            balance=0.2,
            fov_scale=1.0,
        )
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            k_dji,
            d_dji,
            np.eye(3),
            new_k,
            (1920, 1080),
            cv2.CV_16SC2,
        )
        self.homography = np.array(
            [
                [
                    0.8274446551892256,
                    -0.0742944198979625,
                    80.23797348979947,
                ],
                [
                    -0.014725864916652691,
                    0.8471179917075127,
                    28.27366063997317,
                ],
                [
                    -5.083573451500717e-05,
                    -6.846079418201229e-05,
                    1.0,
                ],
            ],
            dtype=np.float64,
        )

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if bgr.shape[:2] != (1080, 1920):
            bgr = cv2.resize(bgr, (1920, 1080), interpolation=cv2.INTER_LINEAR)
        bgr = cv2.remap(bgr, self.map1, self.map2, cv2.INTER_LINEAR)
        bgr = cv2.warpPerspective(bgr, self.homography, (1918, 1105))
        bgr = bgr[115:760, 255:1400]
        bgr = cv2.resize(bgr, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="da3metric-large")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--sequences", nargs="*", default=None)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    from depth_anything_3.api import DepthAnything3

    class AccelerateFP16DepthAnything3(DepthAnything3):
        """Use the official API while leaving autocast to Accelerate."""

        @torch.inference_mode()
        def forward(
            self,
            image,
            extrinsics=None,
            intrinsics=None,
            export_feat_layers=None,
            infer_gs=False,
            use_ray_pose=False,
            ref_view_strategy="saddle_balanced",
        ):
            return self.model(
                image,
                extrinsics,
                intrinsics,
                export_feat_layers,
                infer_gs,
                use_ray_pose,
                ref_view_strategy,
            )

    accelerator = Accelerator(mixed_precision="fp16")
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    sequences = sorted(path for path in data_root.iterdir() if path.is_dir())
    if args.sequences:
        requested = set(args.sequences)
        sequences = [path for path in sequences if path.name in requested]
    local_sequences = sequences[
        accelerator.process_index :: accelerator.num_processes
    ]

    model = AccelerateFP16DepthAnything3(model_name=args.model_name)
    model.load_state_dict(load_file(args.checkpoint, device="cpu"), strict=True)
    model = model.to(accelerator.device).eval()
    calibrate = Calibrator()
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    for sequence in local_sequences:
        rgb = np.load(sequence / "dji_rgb.npy", mmap_mode="r")
        depth_chunks = []
        for start in tqdm(
            range(0, len(rgb), args.batch_size),
            desc=sequence.name,
            disable=not accelerator.is_local_main_process,
        ):
            end = min(start + args.batch_size, len(rgb))
            images = [
                calibrate(np.asarray(rgb[index])) for index in range(start, end)
            ]
            intrinsics = np.repeat(INTRINSICS[None], len(images), axis=0)
            with accelerator.autocast():
                prediction = model.inference(
                    images,
                    intrinsics=intrinsics,
                    process_res=896,
                    process_res_method="upper_bound_resize",
                )
            depth_chunks.append(prediction.depth * SCALE_FACTOR)
        depth = np.concatenate(depth_chunks).astype(np.float32, copy=False)
        np.save(output_dir / f"{sequence.name.lower()}_pred.npy", depth)

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
