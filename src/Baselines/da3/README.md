# Depth Anything 3 inference reference

This folder is the inference-only adapter for the Depth Anything 3 baseline.
It follows the official ByteDance-Seed implementation:

- Repository: <https://github.com/ByteDance-Seed/Depth-Anything-3>
- Pinned reference commit: `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`
- Upstream model: `DA3METRIC-LARGE`
- Upstream API: `depth_anything_3.api.DepthAnything3`

The upstream architecture is installed as a Python package. This artifact does
not duplicate the upstream training, web UI, CLI, or export code. Install the
pinned source before running inference:

```bash
pip install "git+https://github.com/ByteDance-Seed/Depth-Anything-3.git@3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
```

Run the public artifact entry point from `Artifacts_Evaluation/`:

```bash
python src/models/da3/inference.py --gpuid 0
python src/models/da3/inference.py --gpuid 0 2
```

The launcher defaults to the weights-only checkpoint at
`checkpoints/baselines/da3/da3metric-large.safetensors`, reads each
`evaluation_dataset/Smoke-Eval/<sequence>/dji_rgb.npy`, and writes one metric-depth
array per sequence to `inference_results/da3/`. Accelerate controls FP16
autocast and distributes whole sequences when multiple GPU IDs are supplied.
The adapter retains the official input/output pipeline but overrides the
upstream forward wrapper's device-dependent BF16 choice so the requested
Accelerate FP16 policy remains authoritative.

The checkpoint is loaded only through `safetensors.torch.load_file` with strict
key matching. The local camera calibration and scale conversion preserve the
evaluation preprocessing used for the GRADE artifact.

The official source repository and `DA3METRIC-LARGE` model are released under
Apache License 2.0; consult the upstream repository and model card for their
full license terms.
