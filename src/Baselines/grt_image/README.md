# GRT_Image naive ResNet-18 inference backend

This is the paper's naive GRT+Image model: a frozen ResNet-18 image encoder and
GRT radar tokens feed the occupancy decoder. `inference.py` strictly loads the
weights-only `grt_image.safetensors` state dictionary and writes normalized
per-sequence depth arrays. Use `src/models/grt_image/inference.py` as the public
entry point so GPU selection and Accelerate FP16 are applied consistently.
