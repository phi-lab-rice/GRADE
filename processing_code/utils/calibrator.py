"""
Calibrator: defish DJI fisheye → project to ZED-aligned view → crop → resize to ZED resolution.
All parameters are hardcoded in the constructor.
"""

import cv2
import numpy as np


class Calibrator:
    def __init__(self):
        # ZED intrinsics (from SVO, rectified left camera)
        self.K_zed = np.array(
            [
                [521.581604, 0.0, 636.33398438],
                [0.0, 521.581604, 373.10964966],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.zed_w = 1280
        self.zed_h = 720
        # DJI fisheye intrinsics and distortion
        self.K_dji = np.array(
            [
                [718.48555551, 0.0, 963.36465011],
                [0.0, 720.25844189, 537.87569913],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.D_dji = np.array(
            [0.19022699, 0.03466753, 0.05858962, -0.07070669], dtype=np.float64
        )
        self.dji_defish_shape = (1920, 1080)
        self.defish_balance = 0.2
        # Homography from defished DJI to full-extent warped view
        self.H_full_extent = np.array(
            [
                [0.8274446551892256, -0.0742944198979625, 80.23797348979947],
                [-0.014725864916652691, 0.8471179917075127, 28.27366063997317],
                [-5.083573451500717e-05, -6.846079418201229e-05, 1.0],
            ],
            dtype=np.float64,
        )
        self.out_w_full = 1918
        self.out_h_full = 1105
        # Manual crop (top, left, right, bottom)
        self.crop_top = 135 - 20  # 81
        self.crop_left = 255
        self.crop_right = 1400  # 1565
        self.crop_bottom = 780 - 20  # 818
        # Precompute defish maps once (same for all frames)
        R_defish = np.eye(3)
        K_new_defish = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.K_dji,
            self.D_dji,
            self.dji_defish_shape,
            R_defish,
            balance=self.defish_balance,
            fov_scale=1.0,
        )
        self._map1_defish, self._map2_defish = cv2.fisheye.initUndistortRectifyMap(
            self.K_dji,
            self.D_dji,
            R_defish,
            K_new_defish,
            self.dji_defish_shape,
            cv2.CV_16SC2,
        )

    def defish(self, img):
        """Undistort DJI fisheye image (balance=0.2)."""
        return cv2.remap(
            img,
            self._map1_defish,
            self._map2_defish,
            interpolation=cv2.INTER_LINEAR,
        )

    def project(self, img):
        """Warp defished DJI image to ZED-aligned full-extent view."""
        return cv2.warpPerspective(
            img,
            self.H_full_extent,
            (self.out_w_full, self.out_h_full),
            flags=cv2.INTER_LINEAR,
        )

    def crop(self, img):
        """Apply manual pixel crop (top, left, right, bottom)."""
        return img[
            self.crop_top : self.crop_bottom,
            self.crop_left : self.crop_right,
        ]

    def resize(self, img):
        """Resize to ZED resolution (16:9)."""
        return cv2.resize(
            img,
            (self.zed_w, self.zed_h),
            interpolation=cv2.INTER_LINEAR,
        )

    def calibrate(self, img):
        """Full pipeline: defish → project → crop → resize. Returns image in ZED-aligned 16:9."""
        # Ensure input is 1920x1080 (expected DJI resolution)
        if img.shape[1] != 1920 or img.shape[0] != 1080:
            img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_LINEAR)

        img = self.defish(img)
        img1 = self.project(img)
        img2 = self.crop(img1)
        img3 = self.resize(img2)
        return img1, img3


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Load data
    dji_rgb_path = "processed/AlicePratt-2/dji_rgb.npy"
    zed_rgb_path = "processed/AlicePratt-2/zed_rgb.avi"

    print("Loading DJI RGB data...")
    dji_rgb = np.load(dji_rgb_path)
    print(f"DJI RGB shape: {dji_rgb.shape}")

    print("Opening ZED RGB video...")
    cap = cv2.VideoCapture(zed_rgb_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"ZED RGB total frames: {total_frames}")

    # Select a random frame index
    random_idx = np.random.randint(0, min(len(dji_rgb), total_frames))
    print(f"Selected random frame index: {random_idx}")

    # Load DJI frame
    dji_frame = dji_rgb[random_idx]
    print(f"DJI frame shape: {dji_frame.shape}")

    # Load ZED frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_idx)
    ret, zed_frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Failed to read frame {random_idx} from ZED video")

    print(f"ZED frame shape: {zed_frame.shape}")

    # Initialize calibrator and project DJI frame
    print("Applying projection to DJI frame...")
    calibrator = Calibrator()

    # DJI numpy array is RGB, but OpenCV expects BGR - convert before processing
    dji_frame_bgr = cv2.cvtColor(dji_frame, cv2.COLOR_RGB2BGR)

    # Apply full calibration pipeline (defish → project → crop → resize)
    # Returns: img1 (projected), img3 (cropped and resized)
    dji_projected_full, dji_final = calibrator.calibrate(dji_frame_bgr)

    print(f"DJI projected (full extent) shape: {dji_projected_full.shape}")
    print(f"DJI final (cropped & resized) shape: {dji_final.shape}")

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # Convert BGR to RGB for matplotlib
    # ZED frame from VideoCapture is BGR, needs conversion
    zed_rgb_display = cv2.cvtColor(zed_frame, cv2.COLOR_BGR2RGB)
    # DJI frame from numpy array is already RGB, no conversion needed
    dji_rgb_display = dji_frame
    # DJI projected went through OpenCV operations, treat as BGR
    dji_projected_display = cv2.cvtColor(dji_projected_full, cv2.COLOR_BGR2RGB)
    dji_final_display = cv2.cvtColor(dji_final, cv2.COLOR_BGR2RGB)

    axes[0].imshow(zed_rgb_display)
    axes[0].set_title(f"Original ZED RGB (Frame {random_idx})")
    axes[0].axis("off")

    axes[1].imshow(dji_rgb_display)
    axes[1].set_title(f"Original DJI RGB (Frame {random_idx})")
    axes[1].axis("off")

    axes[2].imshow(dji_final_display)
    axes[2].set_title(f"Final Calibrated DJI (Cropped & Resized) (Frame {random_idx})")
    axes[2].axis("off")

    axes[3].imshow(dji_projected_display)
    axes[3].set_title(f"Projected DJI (Full Extent) (Frame {random_idx})")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()

    print("Visualization complete!")
