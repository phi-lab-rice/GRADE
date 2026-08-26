# RadarCam-Depth inference backend

`smoke_eval_inference.py` runs the two-stage RadarCam-Depth baseline: RC-Net
produces quasi-dense depth and SML predicts metric depth. The RC-Net safetensors
file contains `radarnet_encoder.*` and `radarnet_decoder.*` tensors; the SML
file is a plain model state dictionary. Use
`src/models/radarcam-depth/inference.py` as the public entry point so both
weights, Accelerate FP16, and `--gpuid` selection are configured together.
