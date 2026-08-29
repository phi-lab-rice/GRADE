# Source inventory

## Included

- `src/models/`: 16 model-specific inference entry-point folders, one for each
  configuration in `MODELS.md`.
- `src/Baselines/`, `src/GRADE/`, and `src/Ablation/`: shared
  architectures, datasets, and inference implementations used by those entry
  points.
- `checkpoints/`: the 19 weights-only files in `CHECKPOINT_MANIFEST.md`.
- `evaluation/`: archived evaluator inputs, metric computation, CSV merging,
  and paper table/figure reproduction.

## Excluded

- training entry points, checkpoint writers, optimizer/scheduler/loss state,
  W&B metadata, and GRT-CaFNet;
- raw sensor recordings and processed datasets;
- credentials, author/machine-specific paths, bytecode, and repository metadata.

The inference source uses `safetensors.torch.load_file` exclusively for local
model weights. Each public entry point fixes Accelerate mixed precision to FP16
and implements `--gpuid 0` or multi-ID GPU selection.
