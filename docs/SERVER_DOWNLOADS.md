# Box downloads needed for the inference artifact

Release only the canonical camera-ready artifacts through Box. Do not package
historical `latest` checkpoints or W&B run directories.  For every download,
record the source path, byte size, SHA-256, selection metric, and matching
paper model label in `docs/CHECKPOINT_MANIFEST.md` before publication.

## Required data

| Priority | Download | Needed for |
| --- | --- | --- |
| 1 | Processed **Smoke-Eval** test set, including radar inputs, clean RGB where used, ground-truth depth, calibration/metadata, and the exact sequence/frame ordering | all inference and direct metric recomputation |
| 2 | `split.json` matching the camera-ready experiment | validating sequence membership and baseline loaders |
| 3 | Processed Rice / Ti-Zed-Dji release | optional baseline validation and any claimed retraining; not needed for the evaluation-only script |
| 4 | Processed IQ1M release | optional Stage-1 training provenance; not needed for inference with a released checkpoint |

## Required canonical checkpoints

| Paper component | File(s) to retrieve from the server |
| --- | --- |
| GRADE Stage 1 | radar-depth `best_test` checkpoint used by `ours_radar` |
| GRADE Stage 2 | diffusion UNet checkpoint used by `ours_diffusion` |
| GRADE Stage 3 | ControlNet checkpoint used by `ours_full` |
| GRT | validation/test-selected `best` checkpoint |
| GRT+Image | **ResNet-18** `GRT_Image_Naive` validation-selected checkpoint; do not use a DINOv2 checkpoint |
| GRT refinement | frozen/retrained GRT-GRADE Stage-2 and Stage-3 checkpoints |
| CaFNet and CaFNet (No-Smoke) | one selected checkpoint for each paper row |
| RadarCam-Depth | paired RCNet and SML selected checkpoints |
| GRT-CaFNet | selected checkpoint used for the paper row, if this row is reproduced from inference |
| Ablations | selected `ours_radar_no_doppler`, `GRT_no_doppler`, no-gradient, and no-3D artifacts when their paper rows are claimed |

## External immutable dependencies

- `madebyollin/taesd` used by GRADE’s VAE wrapper;
- ImageNet-pretrained ResNet-18 weights for the GRT+Image ResNet constructor;
- any DA3 checkpoint/version used for the camera-only baseline.

Pin revisions and checksums or cache them in the final container.  The current
evaluation-only script does not download any of these; its archived CSV/NPZ
inputs are already included.
