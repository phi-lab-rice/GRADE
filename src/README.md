# Inference-only reference source

This source snapshot contains only the code needed to load the released
weights and run inference. Training entry points and training checkpoints are
not included.

```text
Baselines/
  da3/  grt/  grt_image/  cafnet/  cafnet_no_smoke/  radarcam-depth/
Ablation/
  grt_refine_freeze/  grt_refine_retrain/  ours_radar_no_doppler/
  grt_no_doppler/  ours_radar_no_grad/  ours_full_no_3d/
GRADE/
  stage1_radar_depth_module/  stage2_diffusion_refinement/
```

`models/` contains per-model configurations and compatibility launch adapters.
The supported public model names are defined in
`evaluation/model_registry.py`; use the unified entry point instead of calling
a backend manually:

```bash
python evaluation/run_inference.py --model grade --gpuid 0
python evaluation/run_inference.py --model cafnet --gpuid 0
```

The runner uses one direct Accelerate launch, creates a run-specific YAML, and
resolves every listed checkpoint as a `.safetensors` file. Its `--launcher
wrapper` option retains the original adapter-launch route for debugging.
