# GRADE Artifact Evaluation Package (MobiCom 2026)

This anonymous artifact reproduces the quantitative evaluation of the accepted
MobiCom 2026 GRADE paper. It supports local inference, 2D/3D metric
calculation, and regeneration of the quantitative evaluation tables and
figures. All released model names and generated prediction directories are
lowercase.

## Package layout

```text
Artifacts_Evaluation/
├── environment.txt            # minimal pip environment for this artifact
├── src/                       # inference implementations and model configs
│   ├── Baselines/             # DA3, GRT, CaFNet, and RadarCam-Depth
│   ├── Ablation/              # paper ablation implementations
│   ├── GRADE/                 # Stage 1 radar depth and Stage 2 refinement
│   └── models/                # one release-relative config per model name
├── checkpoints/               # downloaded safetensors weights; empty in release
├── evaluation_dataset/        # downloaded Smoke-Eval inputs; empty in release
├── inference_results/         # archived cluster predictions (reference only)
└── evaluation/
    ├── run_inference.py       # unified model inference runner
    ├── run_metrics.py         # unified 2D -> 3D -> merged-CSV runner
    ├── reproduce_paper.py     # quantitative tables and figures
    ├── outputs/inference/     # newly generated predictions and run manifests
    ├── metric_results/        # metric code and newly generated CSV outputs
    ├── reference_results/     # immutable camera-ready CSVs, tables, figures
    │   ├── paper_figures/     # golden figures, indexed by figure/panel
    │   ├── paper_tables/      # golden Tables 2--7
    │   └── pre_eval_results/  # golden CSV, robustness, and sampling inputs
    └── reproduced_results/    # newly generated tables, figures, comparison HTML
```

`inference_results/` and `evaluation/reference_results/` are the golden
sources. The runners never overwrite them. New files belong only in
`evaluation/outputs/`, `evaluation/metric_results/`, and
`evaluation/reproduced_results/`.

## Download the data and checkpoints

The distributed artifact leaves `evaluation_dataset/` and `checkpoints/`
empty. Download the released payload from
[Hugging Face](https://huggingface.co/datasets/mypersonalsharingspot11/evaluation_dataset),
then place its two top-level directories in this package so that the layout is:

```text
Artifacts_Evaluation/evaluation_dataset/Smoke-Eval/
Artifacts_Evaluation/evaluation_dataset/Smoke-Eval-RadarCam-Depth/
Artifacts_Evaluation/checkpoints/baselines/
Artifacts_Evaluation/checkpoints/grade/
Artifacts_Evaluation/checkpoints/ablations/
Artifacts_Evaluation/checkpoints/third_party/
```

Do not rename the downloaded directories or checkpoints. RadarCam-Depth uses
the separately prepared `Smoke-Eval-RadarCam-Depth` directory; all other
models use `Smoke-Eval`.

## Install the Python environment

Use Python 3.10 or 3.11 and an NVIDIA GPU with a CUDA-compatible PyTorch
wheel. The following example uses a local virtual environment and CUDA 12.6;
change the PyTorch index if the host requires a different CUDA build.

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux/macOS
# source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.7.0 torchvision==0.22.0
python -m pip install -r environment.txt
```

`environment.txt` is a lightweight, pip-installable specification based on
the server's direct dependencies and the imports used by this release. It
intentionally excludes training-only, RAPIDS, notebook, TensorRT, and
cluster-local packages.

## Run inference

From the artifact root, run one canonical model name. The runner launches
Accelerate itself; do not prefix this command with another `accelerate launch`.

```bash
# Full GRADE model (same implementation as ours_full)
python evaluation/run_inference.py --model ours_full --gpuid 0

# A baseline that uses the RadarCam-Depth evaluation dataset
python evaluation/run_inference.py --model radarcam-depth --gpuid 0
```

For multiple GPUs, list the visible GPU IDs, for example
`--gpuid 0 1`. The GRADE diffusion models use the fixed DDIM seed 42. Every
model writes the same layout:

```text
evaluation/outputs/inference/<model>/<lowercase-sequence>_pred.npy
evaluation/outputs/inference/<model>.run.json
evaluation/outputs/inference/.runs/<model>-<timestamp>/<model>.yaml
```

## Compute metrics

After inference, calculate the 2D metrics, 3D metrics, and merged CSV for a
model in one command:

```bash
python evaluation/run_metrics.py --model ours_full
```

By default the metrics runner reads
`evaluation/outputs/inference/<model>/` and writes the per-sequence and
merged CSV results below `evaluation/metric_results/`. To evaluate a different
prediction root explicitly, add `--results-root <path>`. Radar robustness is
optional and intentionally not part of the standard 2D/3D command:

```bash
python evaluation/run_metrics.py --radar-robustness
```

## Reproduce paper tables and figures

After generating the required local merged CSVs, run:

```bash
python evaluation/reproduce_paper.py --mode local
```

This reads `evaluation/metric_results/merged_csv/` and writes Tables 2--7 and
quantitative Figures 9 and 11--14 to `evaluation/reproduced_results/local/`.
Figure 10 is qualitative and intentionally excluded. To regenerate those
artifacts from the saved reference CSVs instead, run:

```bash
python evaluation/reproduce_paper.py --mode saved
```

Saved mode reads `evaluation/reference_results/pre_eval_results/csv/` and
writes its generated artifacts to `evaluation/reproduced_results/saved/`.
Neither mode compares or overwrites golden paper assets.
