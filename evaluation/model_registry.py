"""Shared model names, source locations, and output conventions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InferenceModel:
    """One released model that accepts the packaged Smoke-Eval layout."""

    entrypoint: str
    accelerate_backend: str
    config: str
    metric_variant: str
    prediction_directory: str
    data_key: tuple[str, ...]
    output_key: tuple[str, ...]
    produced_directory: str
    backend_args: tuple[str, ...] = ("--config", "{config}")


# ``grade`` is the paper name for the complete model. The evaluator keeps the
# historical ``ours_full`` merged-CSV name. Backend placeholders are rendered
# by evaluation/run_inference.py after it creates the run-specific YAML.
INFERENCE_MODELS: dict[str, InferenceModel] = {
    # Complete GRADE and its Stage-1/Stage-2 component ablations.
    "grade": InferenceModel(
        "src/models/grade/inference.py",
        "src/GRADE/stage2_diffusion_refinement/inference_full.py",
        "src/models/grade/config.yaml", "ours_full", "ours_full",
        ("data", "smoke_eval_root"), ("inference", "output_root"), "ours_full",
    ),
    "ours_full": InferenceModel(
        "src/models/ours_full/inference.py",
        "src/GRADE/stage2_diffusion_refinement/inference_full.py",
        "src/models/ours_full/config.yaml", "ours_full", "ours_full",
        ("data", "smoke_eval_root"), ("inference", "output_root"), "ours_full",
    ),
    "ours_diffusion": InferenceModel(
        "src/models/ours_diffusion/inference.py",
        "src/GRADE/stage2_diffusion_refinement/inference_diffusion.py",
        "src/models/ours_diffusion/config.yaml", "ours_diffusion", "ours_diffusion",
        ("data", "smoke_eval_root"), ("inference", "output_root"), "ours_diffusion",
    ),
    "ours_radar": InferenceModel(
        "src/models/ours_radar/inference.py",
        "src/GRADE/stage2_diffusion_refinement/inference_radar.py",
        "src/models/ours_radar/config.yaml", "ours_radar", "ours_radar",
        ("data", "smoke_eval_root"), ("inference", "output_root"), "ours_radar",
    ),
    "ours_radar_no_grad": InferenceModel(
        "src/models/ours_radar_no_grad/inference.py",
        # This is a Stage-1-only ablation.  Its wrapper invokes all three
        # GRADE stages, so the direct backend must be the released radar-stage
        # entry point.
        "src/GRADE/stage2_diffusion_refinement/inference_radar.py",
        "src/models/ours_radar_no_grad/config.yaml", "ours_radar_no_grad",
        "ours_radar_no_grad", ("data", "smoke_eval_root"),
        ("inference", "output_root"), "ours_radar",
    ),
    "ours_radar_no_doppler": InferenceModel(
        "src/models/ours_radar_no_doppler/inference.py",
        "src/Ablation/ours_radar_no_doppler/inference.py",
        "src/models/ours_radar_no_doppler/config.yaml", "ours_radar_no_doppler",
        "ours_radar_no_doppler", ("data", "test_root"),
        ("inference", "output_dir"), ".",
    ),
    "ours_full_no_3d": InferenceModel(
        "src/models/ours_full_no_3d/inference.py",
        "src/Ablation/ours_full_no_3d/inference.py",
        "src/models/ours_full_no_3d/config.yaml", "ours_full_no_3d",
        "ours_full_no_3d", ("data", "smoke_eval_root"),
        ("inference", "output_root"), "ours_full",
    ),
    # Baselines.
    "da3": InferenceModel(
        "src/models/da3/inference.py", "src/Baselines/da3/inference.py",
        "src/models/da3/config.yaml", "da3", "da3",
        ("data", "smoke_eval_root"), (), ".",
        (
            "--data_root", "{data_root}",
            "--checkpoint", "{artifact_root}/checkpoints/baselines/da3/da3metric-large.safetensors",
            "--output_dir", "{run_root}",
        ),
    ),
    "grt": InferenceModel(
        "src/models/grt/inference.py", "src/Baselines/grt/inference.py",
        "src/models/grt/config.yaml", "grt", "grt", ("paths", "data_root"), (), ".",
        (
            "--config", "{config}",
            "--checkpoint", "{artifact_root}/checkpoints/baselines/grt/grt.safetensors",
            "--output_dir", "{run_root}",
        ),
    ),
    "grt_image": InferenceModel(
        "src/models/grt_image/inference.py", "src/Baselines/grt_image/inference.py",
        "src/models/grt_image/config.yaml", "grt_image", "grt_image",
        ("paths", "smoke_eval_root"), (), ".",
        (
            "--config", "{config}",
            "--checkpoint", "{artifact_root}/checkpoints/baselines/grt_image/grt_image.safetensors",
            "--output_dir", "{run_root}",
        ),
    ),
    "cafnet": InferenceModel(
        "src/models/cafnet/inference.py", "src/Baselines/cafnet/inference.py",
        "src/models/cafnet/config.yaml", "cafnet", "cafnet",
        ("test_base_dir",), ("prediction_dir",), ".",
    ),
    "cafnet_no_smoke": InferenceModel(
        "src/models/cafnet_no_smoke/inference.py", "src/Baselines/cafnet_no_smoke/inference.py",
        "src/models/cafnet_no_smoke/config.yaml", "cafnet_no_smoke", "cafnet_no_smoke",
        ("test_base_dir",), ("prediction_dir",), ".",
    ),
    "radarcam-depth": InferenceModel(
        "src/models/radarcam-depth/inference.py",
        "src/Baselines/radarcam-depth/smoke_eval_inference.py",
        "src/models/radarcam-depth/config.yaml", "radarcam-depth", "radarcam-depth",
        (), (), ".",
        (
            "--config", "{config}", "--smoke_root", "{data_root}",
            "--rcnet_checkpoint", "{artifact_root}/checkpoints/baselines/radarcam-depth/radarcam-depth_rcnet.safetensors",
            "--sml_checkpoint", "{artifact_root}/checkpoints/baselines/radarcam-depth/radarcam-depth_sml.safetensors",
            "--output_dir", "{run_root}",
        ),
    ),
    # Ablations based on GRT or a modified GRADE loss/input.
    "grt_refine_freeze": InferenceModel(
        "src/models/grt_refine_freeze/inference.py",
        "src/Ablation/grt_refine_freeze/inference_control.py",
        "src/models/grt_refine_freeze/config.yaml", "grt_refine_freeze",
        "grt_refine_freeze", ("data", "smoke_eval_root"),
        ("inference", "output_dir"), ".",
    ),
    "grt_refine_retrain": InferenceModel(
        "src/models/grt_refine_retrain/inference.py",
        "src/Ablation/grt_refine_retrain/inference_control.py",
        "src/models/grt_refine_retrain/config.yaml", "grt_refine_retrain", "grt_refine_retrain",
        ("data", "smoke_eval_root"), ("inference", "output_dir"), ".",
    ),
    "grt_no_doppler": InferenceModel(
        "src/models/grt_no_doppler/inference.py",
        "src/Ablation/grt_no_doppler/inference.py",
        "src/models/grt_no_doppler/config.yaml", "grt_no_doppler",
        "grt_no_doppler", ("paths", "test_data_root"),
        ("inference", "output_dir"), ".",
    ),
}


# Canonical identifiers are used by all folders, raw metric directories, and
# merged CSV filenames.  Old evaluation/paper keys are accepted only when a
# reviewer supplies one explicitly.
CANONICAL_METRIC_VARIANTS = frozenset(
    {
        "da3", "cafnet", "cafnet_no_smoke", "radarcam-depth", "grt",
        "grt_refine_freeze", "grt_no_doppler", "grt_cafnet", "grt_image",
        "grt_refine_retrain", "ours_full", "ours_full_no_3d",
        "ours_diffusion", "ours_radar", "ours_radar_no_doppler",
        "ours_radar_no_grad",
    }
)

MODEL_NAME_ALIASES = {
    "DA3": "da3",
    "CafNet": "cafnet",
    "CafNet_filter": "cafnet_no_smoke",
    "RadarCam_Rice": "radarcam-depth",
    "radarcam_depth": "radarcam-depth",
    "GRT": "grt",
    "GRT_grade_freeze": "grt_refine_freeze",
    "grt_grade": "grt_refine_retrain",
    "GRT_Image": "grt_image",
    "GRT_no_doppler": "grt_no_doppler",
    "ours_radar_no_gradient": "ours_radar_no_grad",
}
# Support lowercased historical spelling such as ``cafnet_filter`` too.
MODEL_NAME_ALIASES.update(
    {
        legacy.lower(): canonical
        for legacy, canonical in tuple(MODEL_NAME_ALIASES.items())
    }
)
EVALUATION_VARIANTS = frozenset(
    {*CANONICAL_METRIC_VARIANTS, *MODEL_NAME_ALIASES}
)


def canonical_model_name(name: str) -> str:
    """Return the canonical artifact identifier for a public/legacy name."""

    return MODEL_NAME_ALIASES.get(name, MODEL_NAME_ALIASES.get(name.lower(), name))


def resolve_metric_variant(name: str) -> str:
    """Translate a public or historical name to the canonical metric name."""

    canonical = canonical_model_name(name)
    if canonical in INFERENCE_MODELS:
        return INFERENCE_MODELS[canonical].metric_variant
    if canonical in CANONICAL_METRIC_VARIANTS:
        return canonical
    choices = sorted({*INFERENCE_MODELS, *EVALUATION_VARIANTS})
    raise ValueError(f"Unknown model {name!r}. Choose from: {', '.join(choices)}")


def artifact_path(root: Path, relative: str) -> Path:
    """Resolve a repository-relative path while keeping registry data portable."""

    return (root / relative).resolve()
