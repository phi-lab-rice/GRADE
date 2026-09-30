# Stage 1: radar-depth module

`radar_depth.py` is the canonical `RadarDepth` implementation loaded from the
GRADE radar checkpoint. Stage 2 imports the same class through its historical
`models.radar_depth` compatibility export, so existing safetensors files retain
their original state-dict key layout.
