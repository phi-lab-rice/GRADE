# GRADE [MobiCom 2026]

Official implementation of **GRADE: Single-Frame Generative Radar Depth Estimation Under Visual Degradation** by Bin Zhao, Patrick Chiou, and Nakul Garg.

[Paper (arXiv)](https://arxiv.org/abs/2609.10756) · [Project website](https://phi-lab-rice.github.io/GRADE/) · [Raw dataset](https://huggingface.co/datasets/phi-lab-rice/GRADE_Dataset) · [Model checkpoints](https://huggingface.co/phi-lab-rice/GRADE) · [BibTeX](#citation)

GRADE estimates metric depth from a single radar frame. A radar depth module provides range information, and a generative refinement module recovers scene structure while using camera cues when they are available.

## Installation

Use Python 3.11. CUDA is recommended for inference and metric computation. Install a PyTorch build compatible with your hardware, then install the dependencies in [environment.txt](environment.txt), which pins the PyTorch and TorchVision versions used for the published results.

```bash
git clone https://github.com/phi-lab-rice/GRADE.git
cd GRADE
conda create -n grade python=3.11
conda activate grade
python -m pip install -r environment.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Inference and evaluation

### Prepare the dataset

Download the [model checkpoints](https://huggingface.co/phi-lab-rice/GRADE) and [raw dataset](https://huggingface.co/datasets/phi-lab-rice/GRADE_Dataset) from Hugging Face. From the repository root, use the provided [processing code](processing_code/) to synchronize the raw training and evaluation recordings into separate processed directories:

```bash
hf download phi-lab-rice/GRADE --include "checkpoints/**" --local-dir .
hf download phi-lab-rice/GRADE --include "evaluation/reference_results/**" --local-dir .
hf download phi-lab-rice/GRADE_Dataset --repo-type dataset --local-dir data/raw
python processing_code/processor.py --dataset data/raw/GRADE_Train_Raw --output-dir data/processed/train
python processing_code/processor.py --dataset data/raw/GRADE_Eval_Raw --output-dir data/processed/eval
```

The processed paths are `data/processed/train/<sequence>/` and `data/processed/eval/<sequence>/`. Each sequence directory contains synchronized `radar.npy` and `dji_rgb.npy` arrays for the current dataset loader, plus `zed_depth.npy` for metric computation. The `data/` directory and downloaded checkpoints are ignored by Git. See the [processor instructions](processing_code/README.md) for input files and output options.

### Run inference locally

Run GRADE on the processed evaluation split, compute per-frame 2D and 3D metrics, then generate tables from the merged CSVs:

```bash
python evaluation/run_inference.py --model grade --data-root data/processed/eval --gpuid 0
python evaluation/run_metrics.py --model grade --data-root data/processed/eval --workers 1
python evaluation/reproduce_paper.py --mode local --tables-only
```

Predictions go to `evaluation/outputs/inference/`, merged metric CSVs to `evaluation/metric_results/merged_csv/`, and tables to `evaluation/reproduced_results/local/tables/`. If you place processed data elsewhere, update `--data-root` in both commands. To compare baselines or ablations, repeat inference and metric computation with their model names; run `python evaluation/run_inference.py --help` for the supported names. Some baselines require additional input layouts described by their code, including RadarCam-Depth. Local tables include only the model results that you computed; paper-wide comparisons require results for all relevant models. Table 7 uses the released sampling-step results unless fresh sampling-step metrics are supplied. Use `--workers 1` for the 3D evaluator.

### Reproduce the paper's numbers from CSVs

To regenerate the paper's quantitative tables without running inference, download the released reference CSVs and supporting results from the [official GRADE model repository](https://huggingface.co/phi-lab-rice/GRADE):

```bash
hf download phi-lab-rice/GRADE --include "evaluation/reference_results/**" --local-dir .
python evaluation/reproduce_paper.py --mode saved --tables-only
```

Tables are written to `evaluation/reproduced_results/saved/tables/`. Omit `--tables-only` to regenerate quantitative figures as well. This path reads the released CSVs and runs no model. The first local LPIPS metric run may download TorchVision's AlexNet weights; cache them before an offline metric run.

## Citation

If you use GRADE, please cite:

```bibtex
@inproceedings{zhao2026grade,
  title     = {GRADE: Single-Frame Generative Radar Depth Estimation Under Visual Degradation},
  author    = {Zhao, Bin and Chiou, Patrick and Garg, Nakul},
  booktitle = {Proceedings of the 32nd Annual International Conference on Mobile Computing and Networking (MobiCom '26)},
  year      = {2026},
  doi       = {10.1145/3795866.3844478}
}
```

## License and acknowledgement

See [LICENSE](LICENSE) for the code license. The project page adapts the [Nerfies](https://nerfies.github.io/) template (CC BY-SA 4.0) and the [RadarSFD](https://github.com/phi-lab-rice/RadarSFD/tree/main/docs) layout.
