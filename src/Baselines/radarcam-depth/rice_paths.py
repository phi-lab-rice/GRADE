"""ZJU-4DRadarCam-style directory layout under a data root.

All scripts build a Layout from cfg.data.output_root (see rice_config.py) so
the on-disk structure is defined in exactly one place.  This module is shared
by radarcam-depth and dataset_prep.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))


class Layout(object):
    def __init__(self, data_root):
        self.data_root = data_root
        self.data_dir = os.path.join(data_root, "data")
        self.result_dir = os.path.join(data_root, "result")
        self.log_dir = os.path.join(data_root, "log")

        self.image_dir = os.path.join(self.data_dir, "image")
        self.radar_npy_dir = os.path.join(self.data_dir, "radar")
        self.radar_png_dir = os.path.join(self.data_dir, "radar_png")
        self.gt_dir = os.path.join(self.data_dir, "gt")
        self.gt_interp_dir = os.path.join(self.data_dir, "gt_interp")

        self.train_list = os.path.join(self.data_dir, "train.txt")
        self.test_list = os.path.join(self.data_dir, "test.txt")
        self.full_list = os.path.join(self.data_dir, "full.txt")

        self.mono_pred_dir = os.path.join(self.result_dir, "mono_pred")
        self.ga_mono_dir = os.path.join(self.result_dir, "global_aligned_mono")
        self.rcnet_result_dir = os.path.join(self.result_dir, "rcnet")

    def ensure_data_dirs(self):
        for d in (
            self.image_dir,
            self.radar_npy_dir,
            self.radar_png_dir,
            self.gt_dir,
            self.gt_interp_dir,
            self.result_dir,
            self.log_dir,
        ):
            os.makedirs(d, exist_ok=True)


def build_layout(data_root):
    return Layout(data_root)
