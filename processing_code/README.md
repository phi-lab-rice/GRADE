# GRADE raw dataset processing

`processor.py` converts raw recordings from the [GRADE dataset](https://huggingface.co/datasets/phi-lab-rice/GRADE_Dataset) into synchronized arrays without the ZED SDK. Its Python dependencies are included in the repository's [environment.txt](../environment.txt).

Each raw sequence needs `radar_*.h5`, `camera_timestamps_*.h5`, `dji_timestamps_*.h5`, `zed_video_anonymized.mkv`, `zed_depth.h5`, and `dji_video_anonymized.mkv`. The camera video and depth file must have matching frame counts and resolution. The processor aligns their timestamps with radar frames.

From the repository root, process the raw training and evaluation directories separately:

```bash
python processing_code/processor.py --dataset data/raw/GRADE_Train_Raw --output-dir data/processed/train
python processing_code/processor.py --dataset data/raw/GRADE_Eval_Raw --output-dir data/processed/eval
```

The result is `data/processed/train/<sequence>/` and
`data/processed/eval/<sequence>/`. To process particular sequences, use
`--sequences` with their directory names:

```bash
python processing_code/processor.py --dataset data/raw/GRADE_Eval_Raw --sequences SEQUENCE_NAME --output-dir data/processed/eval
```

Use `--list-unprocessed` to list sequences without a completed `metadata.json` in the output directory. The processor writes each sequence to `<output-dir>/<sequence>/`. Its main outputs are `radar.npy`, `dji_rgb.npy`, `zed_rgb.npy`, `zed_depth.npy`, `sync_triples.csv`, and `metadata.json`. When present, MAX30105 data become `max30105.npy`; the sensor is optional. Canonical radar point clouds are written as `pcd/pcd_<frame>.npy` by default. `--no-pcd` skips them, and `--no-doppler` writes `radar_no_doppler.npy` instead of `radar.npy`. Run `python processing_code/processor.py --help` for the full set of options.
