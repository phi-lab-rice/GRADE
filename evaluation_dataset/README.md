---
license: cc-by-nc-4.0
---

# Processed evaluation inputs

Use the [official GRADE raw dataset](https://huggingface.co/datasets/phi-lab-rice/GRADE_Dataset)
and the repository's [processing code](../processing_code/) to prepare local
inference inputs. The commands in the [main README](../README.md) write the
training split to `data/processed/train/` and the evaluation split to
`data/processed/eval/`.

The inference and metric scripts expect one subdirectory per sequence under
the processed evaluation root. GRADE inference reads `radar.npy` and
`dji_rgb.npy`; metric computation also reads `zed_depth.npy`. Pass
`--data-root data/processed/eval` to both commands. Processed arrays are ignored by Git.
