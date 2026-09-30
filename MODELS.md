# Model checkpoints and inference code

Every checkpoint in this artifact is a Hugging Face safetensors file containing
only model tensors. Training state (optimizer, scheduler, epoch, losses, and
metrics) is not retained. All model entry points use Hugging Face Accelerate
with FP16 inference in our evaluation

Run any model from the artifact root as follows:

```bash
python src/models/<model>/inference.py --gpuid 0
python src/models/<model>/inference.py --gpuid 0 2
```

The first command uses physical GPU 0. The second exposes physical GPUs 0 and 2
and launches two Accelerate workers. Model-specific arguments can follow the
`--gpuid` values.

## Baselines

| Paper model | Weights | Reference inference folder |
|---|---|---|
| Depth Anything 3 | `checkpoints/baselines/da3/da3metric-large.safetensors` | `src/Baselines/da3/` (launcher: `src/models/da3/`) |
| GRT | `checkpoints/baselines/grt/grt.safetensors` | `src/models/grt/` |
| GRT_Image (naive ResNet-18) | `checkpoints/baselines/grt_image/grt_image.safetensors` | `src/models/grt_image/` |
| CaFNet | `checkpoints/baselines/cafnet/cafnet.safetensors` | `src/models/cafnet/` |
| CaFNet (No-Smoke) | `checkpoints/baselines/cafnet_no_smoke/cafnet_no_smoke.safetensors` | `src/models/cafnet_no_smoke/` |
| RadarCam-Depth | `checkpoints/baselines/radarcam-depth/radarcam-depth_rcnet.safetensors` and `radarcam-depth_sml.safetensors` | `src/models/radarcam-depth/` |
| GRADE (Ours) | `checkpoints/grade/{radar,diffusion,control}.safetensors` | `src/models/grade/` |

## Ablations

| Paper model | Weights | Reference inference folder |
|---|---|---|
| Ours_radar | `checkpoints/grade/radar.safetensors` | `src/models/ours_radar/` |
| Ours_diffusion | `checkpoints/grade/{radar,diffusion}.safetensors` | `src/models/ours_diffusion/` |
| Ours_full (GRADE) | `checkpoints/grade/{radar,diffusion,control}.safetensors` | `src/models/ours_full/` |
| GRT_Refine (Frozen) | baseline GRT plus `checkpoints/grade/{diffusion,control}.safetensors` | `src/models/grt_refine_frozen/` |
| GRT_Refine (Retrain) | `checkpoints/ablations/grt_refine/{grt,diffusion,control}.safetensors` | `src/models/grt_refine_retrain/` |
| Ours_radar w/o Doppler | `checkpoints/ablations/ours_radar_no_doppler/ours_radar_no_doppler.safetensors` | `src/models/ours_radar_no_doppler/` |
| GRT w/o Doppler | `checkpoints/ablations/grt_no_doppler/grt_no_doppler.safetensors` | `src/models/grt_no_doppler/` |
| Ours_radar w/o Grad | `checkpoints/ablations/ours_radar_no_grad/ours_radar_no_grad.safetensors` | `src/models/ours_radar_no_grad/` |
| Ours_full w/o 3D | `checkpoints/ablations/ours_full_no_3d/{radar,diffusion,control}.safetensors` | `src/models/ours_full_no_3d/` |
