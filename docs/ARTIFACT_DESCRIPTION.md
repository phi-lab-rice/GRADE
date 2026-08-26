# Draft artifact description (MobiCom 2026)

## Abstract

This artifact contains the source code and evaluation inputs for GRADE, a
radar-guided depth-estimation system evaluated under clear and smoke-obscured
conditions.  It supports (i) reproducing the paper’s quantitative tables and
figures from archived per-sequence evaluation outputs, and (ii) running
training/inference once the separately released data and canonical checkpoints
are installed.

## Claims evaluated

The primary reproducibility target is the paper’s quantitative evaluation:
Tables 2–7 and Figures 9, 11–13, plus the documented degradation and
robustness analyses.  The evaluation scripts recompute values from stored
per-sequence CSV/NPZ inputs; this target does not require rerunning training.
An end-to-end inference target will be named after final checkpoint provenance
and data-release checks are complete.

## Hardware and software

The released evaluation path requires Python 3.10 or 3.11 and the direct
dependencies in the root `environment.txt`. Inference requires a CUDA-capable
NVIDIA GPU and a PyTorch build matched to the installed driver/CUDA release.
Final submission should record measured GPU memory, RAM, disk, and per-workflow
run-time.

## Installation and execution

Follow the evaluation-only quick start in the package README.  Run
`build_merged.py`, Tables 2–7, and Figures 9/11/12/13.  Each table prints
recomputed values; each figure writes an image below `outputs/`.

For full inference, obtain the exact data versions/checksums listed in
`dataset_prepare/`, configure local release-relative paths, run the model's
inference command, then run `evaluation/metric_results/run_evaluation.py` to
compute 2D and 3D metrics and rebuild the merged evaluation results.

## Expected results and limitations

Expected numerical results and accepted tolerances will be completed from a
clean-container validation log. Raw data are not included in this preparation
snapshot; `checkpoints/` contains only the explicitly selected GRADE, GRT, and
no-3D-loss weights. The artifact must not claim full retraining
reproducibility until those components are public or the AE committee has an
agreed anonymous remote-access workflow.
