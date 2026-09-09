#!/usr/bin/env python3
"""Smoke-Eval inference: RC-Net -> SML metric depth, one sequence at a time.

Runs the complete RadarCam-Depth inference chain (RC-Net quasi-dense depth ->
Scale Map Learner metric depth) over *every* frame of *every* Smoke-Eval
sequence and writes one file per sequence:

    <output_dir>/<sequence>_pred.npy    float32 [N, 1, H, W], depth in METERS

Nothing has to be pointed at by hand: the script discovers

  * the prepared Smoke-Eval root (ZJU-style layout, see rice_paths.Layout),
  * the weights-only RC-Net safetensors file,
  * the weights-only SML safetensors file,

and each of them can still be overridden from the command line.

DDP and configured mixed precision come from HuggingFace Accelerate. Every
rank takes a disjoint stride of the current sequence's frames. Per-sequence
predictions are gathered on the main process, deduplicated by frame index and
checked for completeness -- a sequence file is written only when all of its
frames are present exactly once.

The Smoke-Eval data must already be prepared into the ZJU layout with
../dataset_prep (image / radar / gt + global-aligned mono depth), exactly like
the training data: SML consumes the globally aligned monocular depth, which is
produced there and not by this script.

Usage:
    accelerate launch smoke_eval_inference.py
    accelerate launch smoke_eval_inference.py --smoke_root /path/to/smoke_data
    python smoke_eval_inference.py --config config.yaml --output_dir preds
"""

import argparse
import contextlib
import os
import pickle
import re
from collections import OrderedDict

import numpy as np
import torch
import torch.utils.data
from accelerate import Accelerator
from accelerate.utils import set_seed
from safetensors.torch import load_file
from tqdm.auto import tqdm

import data.data_utils as data_utils
import modules.midas.transforms as sml_transforms
import modules.midas.utils as midas_utils
import rice_paths
from modules.midas.midas_net_custom import MidasNet_small_videpth
from rcnet_inference import forward_with_fallback
from rcnet_model import RCNetModel
from rcnet_transforms import Transforms
from rice_config import load_config

# Environment overrides, in the spirit of rice_config's RICE_DATA_ROOT.
SMOKE_ROOT_ENV = "SMOKE_EVAL_ROOT"

# We distribute weights-only safetensors files, not training checkpoints.
CHECKPOINT_PREFERENCE = ("model.safetensors",)

# "<sequence>_<frame>" (the dataset_prep naming) or "<sequence>/<frame>".
# The greedy sequence group makes the *last* numeric field the frame index,
# so sequence names may themselves contain digits, '-' and '_'.
DEFAULT_NAME_PATTERN = r"^(?P<seq>.+)[_/](?P<frame>\d+)$"

DEFAULT_OUTPUT_DIRNAME = "prediction_smoke_eval"


# ---------------------------------------------------------------------------
# Discovery: dataset root and checkpoints
# ---------------------------------------------------------------------------


def _mono_ga_dirpath(layout, mono_tag):
    return os.path.join(layout.ga_mono_dir, mono_tag + "_ls")


def _is_prepared_root(path, mono_tag):
    """A prepared ZJU-layout root has the inputs both stages need."""
    if not path or not os.path.isdir(path):
        return False
    layout = rice_paths.build_layout(path)
    required = [
        layout.image_dir,
        layout.radar_npy_dir,
        _mono_ga_dirpath(layout, mono_tag),
    ]
    return all(os.path.isdir(d) for d in required)


def _search_bases(cfg):
    """Directories that plausibly hold a prepared Smoke-Eval root."""
    output_root = cfg.data.output_root
    bases = [
        rice_paths.HERE,
        os.path.dirname(rice_paths.HERE),
        output_root,
        os.path.dirname(output_root),
        os.path.dirname(os.path.dirname(output_root)),
        os.getcwd(),
    ]
    unique = []
    for base in bases:
        base = os.path.abspath(base)
        if base not in unique:
            unique.append(base)
    return unique


def _auto_candidates(cfg):
    """Candidate roots: any 'smoke'-named directory near the project/data."""
    candidates = []
    for base in _search_bases(cfg):
        if not os.path.isdir(base):
            continue
        try:
            children = sorted(os.listdir(base))
        except OSError:
            continue
        for child in children:
            if "smoke" not in child.lower():
                continue
            path = os.path.join(base, child)
            if os.path.isdir(path) and path not in candidates:
                candidates.append(path)
    return candidates


def _resolve_candidate(path, mono_tag):
    """Accept the candidate itself or a single prepared root nested in it."""
    if _is_prepared_root(path, mono_tag):
        return os.path.abspath(path)
    if os.path.isdir(path):
        for child in sorted(os.listdir(path)):
            nested = os.path.join(path, child)
            if _is_prepared_root(nested, mono_tag):
                return os.path.abspath(nested)
    return None


def discover_smoke_root(cfg, explicit, mono_tag):
    """Locate the prepared Smoke-Eval root.

    Priority: --smoke_root, $SMOKE_EVAL_ROOT, data.smoke_eval_root in the YAML,
    then a scan for 'smoke'-named directories beside the project and the
    prepared training data.
    """
    explicit_sources = [
        (explicit, "--smoke_root"),
        (os.environ.get(SMOKE_ROOT_ENV), "${}".format(SMOKE_ROOT_ENV)),
        (cfg.data.get("smoke_eval_root"), "data.smoke_eval_root in the config"),
    ]
    for path, origin in explicit_sources:
        if not path:
            continue
        resolved = _resolve_candidate(path, mono_tag)
        if resolved is None:
            raise FileNotFoundError(
                "{} points at '{}', which is not a prepared Smoke-Eval root "
                "(expected data/image, data/radar and "
                "result/global_aligned_mono/{}_ls underneath it).".format(
                    origin, path, mono_tag
                )
            )
        return resolved

    for candidate in _auto_candidates(cfg):
        resolved = _resolve_candidate(candidate, mono_tag)
        if resolved is not None:
            return resolved

    raise FileNotFoundError(
        "Could not find a prepared Smoke-Eval root.  Looked for directories "
        "with 'smoke' in their name under:\n  {}\n"
        "A prepared root contains data/image, data/radar and "
        "result/global_aligned_mono/{}_ls (produce it with "
        "../dataset_prep/prepare_all.py on the Smoke-Eval recordings).  "
        "Pass --smoke_root, set ${}, or add data.smoke_eval_root to the "
        "config to point at it explicitly.".format(
            "\n  ".join(_search_bases(cfg)), mono_tag, SMOKE_ROOT_ENV
        )
    )


def discover_checkpoint(save_dir, explicit, label):
    """Locate the best checkpoint for one stage.

    Priority: the explicit safetensors path, then the configured save directory.
    """
    if explicit:
        path = explicit if os.path.isabs(explicit) else os.path.join(rice_paths.HERE, explicit)
        if os.path.isfile(path):
            return os.path.abspath(path)
        if os.path.isdir(path):
            save_dir = path
        else:
            raise FileNotFoundError("{} checkpoint not found: {}".format(label, explicit))

    if not os.path.isabs(save_dir):
        save_dir = os.path.join(rice_paths.HERE, save_dir)

    if not os.path.isdir(save_dir):
        raise FileNotFoundError(
            "{} checkpoint directory not found: {}".format(label, save_dir)
        )

    for name in CHECKPOINT_PREFERENCE:
        path = os.path.join(save_dir, name)
        if os.path.isfile(path):
            return os.path.abspath(path)

    available = sorted(
        f for f in os.listdir(save_dir) if f.endswith(".safetensors")
    )
    raise FileNotFoundError(
        "No usable {} checkpoint in {} (looked for {}; found {}).".format(
            label, save_dir, ", ".join(CHECKPOINT_PREFERENCE), available or "none"
        )
    )


# ---------------------------------------------------------------------------
# Frame bookkeeping
# ---------------------------------------------------------------------------


def load_names(layout, explicit_list):
    """Every frame name of the Smoke-Eval root, in file order, deduplicated."""
    if explicit_list:
        list_path = explicit_list
    else:
        # full.txt covers every prepared frame; test.txt is the fallback when
        # the prep wrote only a split.  Otherwise read the image directory.
        list_path = None
        for candidate in (layout.full_list, layout.test_list, layout.train_list):
            if os.path.isfile(candidate):
                list_path = candidate
                break

    if list_path is not None:
        if not os.path.isfile(list_path):
            raise FileNotFoundError("Frame list not found: {}".format(list_path))
        with open(list_path, "r") as f:
            names = [line.strip() for line in f if line.strip()]
    else:
        if not os.path.isdir(layout.image_dir):
            raise FileNotFoundError("Image directory not found: {}".format(layout.image_dir))
        names = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(layout.image_dir)
            if f.endswith(".png")
        )

    if not names:
        raise ValueError("No Smoke-Eval frames found (source: {}).".format(list_path or layout.image_dir))

    seen = set()
    unique = []
    for name in names:
        if name not in seen:
            seen.add(name)
            unique.append(name)
    return unique, (list_path or layout.image_dir)


def group_by_sequence(names, pattern):
    """Map frame names to {sequence: [(frame_idx, name), ...]} sorted by frame."""
    regex = re.compile(pattern)
    grouped = OrderedDict()
    unmatched = []

    for name in names:
        match = regex.match(name)
        if match is None:
            unmatched.append(name)
            continue
        seq = match.group("seq")
        frame_idx = int(match.group("frame"))
        grouped.setdefault(seq, {})
        # Duplicate frame ids inside a sequence keep the first occurrence.
        grouped[seq].setdefault(frame_idx, name)

    if unmatched:
        raise ValueError(
            "{} frame name(s) do not match the sequence/frame pattern '{}', "
            "e.g. {}.  Pass --name_pattern with named groups 'seq' and "
            "'frame'.".format(len(unmatched), pattern, unmatched[:5])
        )

    ordered = OrderedDict()
    for seq in sorted(grouped):
        ordered[seq] = [(idx, grouped[seq][idx]) for idx in sorted(grouped[seq])]
    return ordered


def _safe_name(seq_name):
    return seq_name.replace("/", "_").replace("\\", "_").lower()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SmokeEvalFrames(torch.utils.data.Dataset):
    """Per-frame inputs for the RC-Net + SML chain, from the prepared layout.

    Returns raw arrays; the geometry-dependent parts (bounding boxes, scale
    map) are built in the inference loop, where the RC-Net output is known.
    """

    def __init__(self, entries, layout, mono_ga_dir, load_gt):
        self.entries = entries
        self.image_dir = layout.image_dir
        self.radar_dir = layout.radar_npy_dir
        self.gt_dir = layout.gt_dir if load_gt else None
        self.mono_ga_dir = mono_ga_dir

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        frame_idx, name = self.entries[index]

        # 0-255 HWC float; RC-Net's Transforms normalizes it, SML wants [0, 1].
        image = data_utils.load_image(
            os.path.join(self.image_dir, name + ".png"),
            normalize=False,
            data_format="HWC",
        ).astype(np.float32)

        radar_points = np.load(os.path.join(self.radar_dir, name + ".npy"))
        radar_points = np.asarray(radar_points, dtype=np.float32)
        if radar_points.ndim == 1:
            radar_points = np.expand_dims(radar_points, axis=0)
        if radar_points.size == 0:
            radar_points = np.zeros((0, 3), dtype=np.float32)

        mono_ga = _load_metric_depth_png(os.path.join(self.mono_ga_dir, name + ".png"))
        mono_ga = np.clip(mono_ga, 1e-3, None)

        sample = {
            "frame_idx": int(frame_idx),
            "name": name,
            "image": image,
            "radar_points": radar_points,
            "mono_ga": mono_ga,
        }

        if self.gt_dir is not None:
            sample["gt"] = _load_metric_depth_png(
                os.path.join(self.gt_dir, name + ".png")
            )
        return sample


def _load_metric_depth_png(path):
    """16-bit depth PNG -> float32 meters (the data_utils encoding)."""
    return data_utils.load_depth(path, data_format="HW").astype(np.float32)


def _identity_collate(batch):
    """Radar point counts vary per frame, so keep the batch as a list."""
    return batch


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def _strip_module_prefix(state_dict):
    return {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }


def build_rcnet(cfg, checkpoint_path, device):
    rc = cfg.rcnet
    model = RCNetModel(
        input_channels_image=3,
        input_channels_depth=3,
        input_patch_size_image=rc.patch_size,
        encoder_type=rc.encoder_type,
        n_filters_encoder_image=rc.n_filters_encoder_image,
        n_neurons_encoder_depth=rc.n_neurons_encoder_depth,
        decoder_type=rc.decoder_type,
        n_filters_decoder=rc.n_filters_decoder,
        weight_initializer=rc.weight_initializer,
        activation_func=rc.activation_func,
        device=device,
    )

    checkpoint = load_file(checkpoint_path, device="cpu")
    encoder_prefix = "radarnet_encoder."
    decoder_prefix = "radarnet_decoder."
    encoder_state = {
        key[len(encoder_prefix):]: value
        for key, value in checkpoint.items()
        if key.startswith(encoder_prefix)
    }
    decoder_state = {
        key[len(decoder_prefix):]: value
        for key, value in checkpoint.items()
        if key.startswith(decoder_prefix)
    }
    if not encoder_state or not decoder_state:
        raise ValueError(
            "Unsupported RC-Net checkpoint format: {}".format(checkpoint_path)
        )
    # The trainer stores 'module.'-prefixed keys for baseline compatibility;
    # each DDP rank here runs an unwrapped copy, so strip the prefix.
    model.encoder.load_state_dict(
        _strip_module_prefix(encoder_state)
    )
    model.decoder.load_state_dict(
        _strip_module_prefix(decoder_state)
    )
    model.eval()
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def build_sml(cfg, checkpoint_path, accelerator):
    # The efficientnet backbone comes from torch.hub on first use; serialize
    # the download so DDP ranks do not race for the cache.
    with accelerator.main_process_first():
        model = MidasNet_small_videpth(
            device="cpu",
            min_pred=cfg.depth.min_pred_depth_m,
            max_pred=cfg.depth.max_pred_depth_m,
        )
    # BaseModel.load() understands the trainer's {"model": ...} payload.
    model.load(checkpoint_path)
    model.eval()
    model.to(accelerator.device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def rcnet_quasi_dense(model, transforms, image_hwc, radar_points, patch_size,
                      response_thr, device):
    """RC-Net quasi-dense depth for one frame -> float32 [H, W] in meters."""
    height, width = image_hwc.shape[:2]

    if radar_points.shape[0] == 0:
        # No radar returns for this frame: SML still runs, on a scale map with
        # no anchors.  The frame is kept so the sequence stays complete.
        return np.zeros((height, width), dtype=np.float32)

    image = torch.from_numpy(np.transpose(image_hwc, (2, 0, 1))).unsqueeze(0).to(device)
    points = torch.from_numpy(radar_points).to(device)

    # Same boxes as rcnet_inference.run: full-height patches centered on each
    # radar point in the horizontally padded image.  Built on the model device
    # because torchvision.ops.roi_pool does not move them for us.
    pad_size_x = patch_size[1] // 2
    points[:, 0] = points[:, 0] + pad_size_x
    x = points[:, 0]
    bounding_boxes = torch.stack(
        [
            x - pad_size_x,
            torch.zeros_like(x),
            x + pad_size_x,
            torch.full_like(x, float(height)),
        ],
        dim=1,
    )
    bounding_boxes_list = [bounding_boxes]

    [image], [points], [bounding_boxes_list] = transforms.transform(
        images_arr=[image],
        points_arr=[points],
        bounding_boxes_arr=[bounding_boxes_list],
        random_transform_probability=0.0,
    )

    output_depth, _, _, inference_failed = forward_with_fallback(
        model=model,
        image=image,
        radar_points=points,
        bounding_boxes_list=bounding_boxes_list,
        response_thr=response_thr,
        device=device,
    )

    if inference_failed:
        # Keep going: an all-zero quasi-dense map degrades SML to the
        # globally aligned mono depth for this frame rather than dropping it.
        return np.zeros((height, width), dtype=np.float32)

    return np.squeeze(output_depth.float().cpu().numpy()).astype(np.float32)


def build_sml_sample(image_hwc, rcnet_depth, mono_ga, transform, depth_cfg):
    """Scale-map inputs for one frame, matching sml_train_rice's math."""
    rcnet_valid = (rcnet_depth < depth_cfg.max_radar_depth_m) & (
        rcnet_depth > depth_cfg.min_radar_depth_m
    )

    int_depth = (1.0 / mono_ga).astype(np.float32)
    int_scales = np.ones_like(int_depth)
    int_scales[rcnet_valid] = (1.0 / rcnet_depth[rcnet_valid]) / int_depth[rcnet_valid]
    if np.ptp(int_scales) > 0:
        int_scales = midas_utils.normalize_unit_range(int_scales.astype(np.float32))
    else:
        # No quasi-dense anchors in this frame: constant mid-range scale.
        int_scales = np.full_like(int_depth, 0.5)

    sample = {
        "image": (image_hwc / 255.0).astype(np.float32),
        "int_depth": int_depth,
        "int_scales": int_scales,
        "int_depth_no_tf": int_depth,
    }
    sample = transform(sample)
    x = torch.cat([sample["int_depth"], sample["int_scales"]], 0)
    return x, sample["int_depth_no_tf"]


def run_sequence(
    seq_name,
    entries,
    layout,
    mono_ga_dir,
    rcnet_model,
    sml_model,
    rcnet_transforms,
    sml_transform,
    cfg,
    accelerator,
    args,
    output_dir,
    gather_dir,
):
    """Infer one sequence on all ranks and save <sequence>_pred.npy on rank 0."""
    device = accelerator.device
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    # Disjoint stride per rank: every frame is processed exactly once.
    rank_entries = entries[rank::world_size]

    compute_metrics = (not args.no_metrics) and os.path.isdir(layout.gt_dir)

    # RC-Net pools ROIs with torchvision.ops.roi_pool, which only has an
    # autocast kernel for CUDA (it casts feature map and boxes back to fp32
    # there, as during training).  On CPU/MPS autocast would feed the kernel a
    # half-precision feature map with fp32 boxes, so keep that stage in fp32.
    rcnet_autocast = (
        accelerator.autocast if device.type == "cuda" else contextlib.nullcontext
    )

    loader = torch.utils.data.DataLoader(
        SmokeEvalFrames(rank_entries, layout, mono_ga_dir, load_gt=compute_metrics),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=_identity_collate,
    )

    local_predictions = []  # (frame_idx, [1, H, W] float32)
    abs_err_sum = 0.0
    sq_err_sum = 0.0
    n_valid_px = 0.0

    progress = tqdm(
        loader,
        desc="  {}".format(seq_name),
        disable=not accelerator.is_local_main_process,
        leave=False,
    )

    for batch in progress:
        batch_x = []
        batch_d = []

        for sample in batch:
            with rcnet_autocast():
                rcnet_depth = rcnet_quasi_dense(
                    model=rcnet_model,
                    transforms=rcnet_transforms,
                    image_hwc=sample["image"],
                    radar_points=sample["radar_points"],
                    patch_size=cfg.rcnet.patch_size,
                    response_thr=args.response_thr,
                    device=device,
                )
            x, d = build_sml_sample(
                image_hwc=sample["image"],
                rcnet_depth=rcnet_depth,
                mono_ga=sample["mono_ga"],
                transform=sml_transform,
                depth_cfg=cfg.depth,
            )
            batch_x.append(x)
            batch_d.append(d)

        x = torch.stack(batch_x, dim=0).to(device)
        d = torch.stack(batch_d, dim=0).to(device)

        with accelerator.autocast():
            pred_inv, _ = sml_model(x, d)

        # Metric depth at the prepared resolution, as in sml_inference.validate.
        height, width = batch[0]["image"].shape[:2]
        pred_depth = torch.nn.functional.interpolate(
            1.0 / pred_inv.float(),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        )
        pred_np = pred_depth.detach().cpu().numpy().astype(np.float32)

        for i, sample in enumerate(batch):
            frame_pred = pred_np[i]  # [1, H, W], meters
            local_predictions.append((sample["frame_idx"], frame_pred))

            if compute_metrics:
                gt = sample["gt"]
                mask = (
                    (gt > 0)
                    & (gt > cfg.depth.min_eval_depth_m)
                    & (gt < cfg.depth.max_eval_depth_m)
                )
                if np.any(mask):
                    error = frame_pred[0][mask] - gt[mask]
                    abs_err_sum += float(np.abs(error).sum())
                    sq_err_sum += float((error ** 2).sum())
                    n_valid_px += float(mask.sum())

    # Gather this sequence through per-rank files: bounded memory, and no
    # object-collective large enough to matter.
    accelerator.wait_for_everyone()
    safe_seq = _safe_name(seq_name)
    rank_file = os.path.join(gather_dir, "rank_{}_{}.pkl".format(rank, safe_seq))
    with open(rank_file, "wb") as f:
        pickle.dump(local_predictions, f, protocol=pickle.HIGHEST_PROTOCOL)

    metrics_local = torch.tensor(
        [abs_err_sum, sq_err_sum, n_valid_px],
        device=device,
        dtype=torch.float64,
    )
    metrics_global = accelerator.reduce(metrics_local, reduction="sum")
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        merged = []
        for r in range(world_size):
            path = os.path.join(gather_dir, "rank_{}_{}.pkl".format(r, safe_seq))
            with open(path, "rb") as f:
                merged.extend(pickle.load(f))
            os.remove(path)

        # Deduplicate by frame index (keep first) and order by frame.
        by_frame = {}
        for frame_idx, pred in merged:
            if frame_idx not in by_frame:
                by_frame[frame_idx] = pred

        expected = [frame_idx for frame_idx, _ in entries]
        missing = [frame_idx for frame_idx in expected if frame_idx not in by_frame]
        if missing:
            raise RuntimeError(
                "{}: missing {} prediction(s), first few frame indices: "
                "{}".format(seq_name, len(missing), missing[:5])
            )

        pred_stack = np.stack(
            [by_frame[frame_idx] for frame_idx in sorted(by_frame)], axis=0
        ).astype(np.float32)

        if pred_stack.ndim != 4 or pred_stack.shape[1] != 1:
            raise RuntimeError(
                "{}: expected [N, 1, H, W], got {}".format(seq_name, pred_stack.shape)
            )
        if pred_stack.shape[0] != len(expected):
            raise RuntimeError(
                "{}: expected {} frames, got {}".format(
                    seq_name, len(expected), pred_stack.shape[0]
                )
            )
        if not np.isfinite(pred_stack).all():
            raise RuntimeError("{}: predictions contain NaN or Inf".format(seq_name))

        out_path = os.path.join(output_dir, "{}_pred.npy".format(safe_seq))
        np.save(out_path, pred_stack)

        message = "  saved {} frames  shape={}  range=[{:.2f}, {:.2f}] m  -> {}".format(
            pred_stack.shape[0],
            tuple(pred_stack.shape),
            float(pred_stack.min()),
            float(pred_stack.max()),
            out_path,
        )
        if compute_metrics and float(metrics_global[2].item()) > 0:
            count = float(metrics_global[2].item())
            mae = float(metrics_global[0].item()) / count
            rmse = (float(metrics_global[1].item()) / count) ** 0.5
            message += "\n  MAE={:.4f} m  RMSE={:.4f} m".format(mae, rmse)
        print(message, flush=True)

    accelerator.wait_for_everyone()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=os.path.join(rice_paths.HERE, "config.yaml"),
        help="Path to the nested YAML config",
    )
    parser.add_argument(
        "--smoke_root",
        default="",
        help="Prepared Smoke-Eval root (default: auto-discovered)",
    )
    parser.add_argument(
        "--rcnet_checkpoint",
        default="",
        help="RC-Net .safetensors file",
    )
    parser.add_argument(
        "--sml_checkpoint",
        default="",
        help="SML .safetensors file",
    )
    parser.add_argument(
        "--output_dir",
        default="",
        help="Where to write <sequence>_pred.npy (default: ./{})".format(
            DEFAULT_OUTPUT_DIRNAME
        ),
    )
    parser.add_argument(
        "--list",
        default="",
        help="Frame-name list (default: the root's full.txt, else test.txt, "
        "else every image in data/image)",
    )
    parser.add_argument(
        "--name_pattern",
        default=DEFAULT_NAME_PATTERN,
        help="Regex with named groups 'seq' and 'frame' for frame names",
    )
    parser.add_argument("--batch_size", type=int, default=None, help="Per-GPU SML batch")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--response_thr", type=float, default=None)
    parser.add_argument(
        "--no_metrics",
        action="store_true",
        help="Skip the MAE/RMSE sanity check against the prepared ground truth",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.batch_size is None:
        args.batch_size = cfg.sml.batch_size
    if args.num_workers is None:
        args.num_workers = cfg.sml.num_workers
    if args.response_thr is None:
        args.response_thr = cfg.rcnet.response_thr
    mixed_precision = cfg.runtime.mixed_precision

    smoke_root = discover_smoke_root(cfg, args.smoke_root, cfg.sml.mono_tag)
    layout = rice_paths.build_layout(smoke_root)
    mono_ga_dir = _mono_ga_dirpath(layout, cfg.sml.mono_tag)

    rcnet_checkpoint = discover_checkpoint(
        cfg.rcnet.save_dir, args.rcnet_checkpoint, "RC-Net"
    )
    sml_checkpoint = discover_checkpoint(cfg.sml.save_dir, args.sml_checkpoint, "SML")

    output_dir = args.output_dir or os.path.join(rice_paths.HERE, DEFAULT_OUTPUT_DIRNAME)
    output_dir = os.path.abspath(output_dir)

    names, names_source = load_names(layout, args.list)
    sequences = group_by_sequence(names, args.name_pattern)

    set_seed(cfg.runtime.seed)
    accelerator = Accelerator(mixed_precision=mixed_precision, cpu=cfg.runtime.cpu)

    gather_dir = os.path.join(output_dir, "_gather")
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(gather_dir, exist_ok=True)
        print("Smoke-Eval root   : {}".format(smoke_root))
        print("Frame list        : {}".format(names_source))
        print("RC-Net checkpoint : {}".format(rcnet_checkpoint))
        print("SML checkpoint    : {}".format(sml_checkpoint))
        print("Output directory  : {}".format(output_dir))
        print(
            "Sequences: {} | frames: {} | processes: {} | mixed precision: {}".format(
                len(sequences), len(names), accelerator.num_processes, mixed_precision
            )
        )
    accelerator.wait_for_everyone()

    rcnet_model = build_rcnet(cfg, rcnet_checkpoint, accelerator.device)
    sml_model = build_sml(cfg, sml_checkpoint, accelerator)
    rcnet_transforms = Transforms(
        normalized_image_range=cfg.rcnet.normalized_image_range
    )
    sml_transform = sml_transforms.get_transforms(cfg.sml.mono_tag, "void", "150")

    with torch.no_grad():
        for seq_idx, (seq_name, entries) in enumerate(sequences.items()):
            if accelerator.is_main_process:
                print(
                    "\n[{}/{}] {}  ({} frames)".format(
                        seq_idx + 1, len(sequences), seq_name, len(entries)
                    ),
                    flush=True,
                )
            run_sequence(
                seq_name=seq_name,
                entries=entries,
                layout=layout,
                mono_ga_dir=mono_ga_dir,
                rcnet_model=rcnet_model,
                sml_model=sml_model,
                rcnet_transforms=rcnet_transforms,
                sml_transform=sml_transform,
                cfg=cfg,
                accelerator=accelerator,
                args=args,
                output_dir=output_dir,
                gather_dir=gather_dir,
            )

    if accelerator.is_main_process:
        if os.path.isdir(gather_dir) and not os.listdir(gather_dir):
            os.rmdir(gather_dir)
        print("\nInference complete: {} sequence file(s) in {}".format(
            len(sequences), output_dir
        ))


if __name__ == "__main__":
    main()
