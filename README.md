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

The artifact-evaluation package — inference code, evaluation harness, and model weights — will be
released here after artifact evaluation concludes.

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
