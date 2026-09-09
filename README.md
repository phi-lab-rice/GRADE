# GRADE

**Single-Frame Generative Radar Depth Estimation Under Visual Degradation**

Bin Zhao, Patrick Chiou, Nakul Garg — Rice University
ACM MobiCom 2026 · Austin, TX

Dense 3D depth perception fails under smoke, fog, and darkness because optical sensors cannot
penetrate airborne particulates. mmWave radar works in these conditions but its limited angular
resolution gives depth that is metrically grounded yet structurally incomplete. GRADE grounds
pretrained generative priors in single-frame radar geometry to recover high-fidelity metric depth —
without SAR and without a reliable camera.

Trained and evaluated on ~95K synchronized frames across 12 buildings with real smoke using
leave-building-out splits, GRADE reaches an MAE of 0.303 m in clear conditions and 0.313 m under
smoke, ahead of every baseline on all reported metrics.

## Project page

<https://phi-lab-rice.github.io/GRADE/>

Served from [`docs/`](docs/) via GitHub Pages.

## Code and artifacts

The evaluation code is included directly in this repository under
[`evaluation/`](evaluation/) and [`src/`](src/). It includes inference, metric
computation, saved-result reproduction, and the evaluation configuration.

### Dataset

The synchronized dataset is shared through
[Box](https://rice.box.com/s/8rfrycts3cxzpc3g0h68d33hg6u98t2p). Please follow the
access and usage terms provided with that share link. The dataset-processing
scripts are available directly in [`processing_code/`](processing_code/) in this
GitHub repository and under `processing_code/` in the Box share.

The Box release includes the model checkpoints and is organized as follows:

```text
GRADE Dataset/
├── checkpoints/      # released model checkpoints
├── processing_code/  # dataset-processing scripts
├── GRADE_Eval_Raw/   # evaluation data
└── GRADE_Train_Raw/  # training data
```

Download the `checkpoints/` directory together with the dataset when running
full evaluation. The evaluation scripts use the checkpoint paths configured in
the model registry and related configuration files.

To use the processing scripts locally from this repository:

```bash
cd processing_code

# Full radar + ZED + DJI processing
python processor.py --dataset /path/to/raw_dataset

# RGB/depth-only processing
python processor_rgb.py --dataset /path/to/raw_dataset

# Radar point-cloud extraction
python processor_pcd.py --dataset /path/to/raw_dataset
```

Each processor accepts `--sequences` to process selected sequences. The
generated files are written under `processed/<sequence_name>/`, including
synchronized timestamps and the processed radar, RGB, depth, or point-cloud
outputs appropriate to the selected pipeline. See the docstrings in the
processing scripts for optional modality skips and split-file arguments.

For reproducible evaluation, install the dependencies from
[`environment.txt`](environment.txt) before running the evaluation code. A
typical setup is:

```bash
python3.11 -m venv grade-venv
source grade-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r environment.txt
```

After preparing the environment, follow the command-line help and docstrings
in the scripts under [`evaluation/`](evaluation/). Full evaluation also
requires downloading the checkpoints and the required dataset directories from
the Box release.

## Citation

```bibtex
@inproceedings{zhao2026grade,
  title     = {GRADE: Single-Frame Generative Radar Depth Estimation Under Visual Degradation},
  author    = {Zhao, Bin and Chiou, Patrick and Garg, Nakul},
  booktitle = {Proceedings of the 32nd Annual International Conference on
               Mobile Computing and Networking (MobiCom '26)},
  year      = {2026},
  doi       = {10.1145/3795866.3844478}
}
```

## Acknowledgement

The project page is based on the [Nerfies](https://nerfies.github.io/) template
(CC BY-SA 4.0), with the layout adapted from our
[RadarSFD project page](https://github.com/phi-lab-rice/RadarSFD/tree/main/docs).
