"""
Helper to convert depth images (millimeters) to point clouds (meters)
using Open3D and the same pinhole camera model as pc_offiline.py
(verified accuracy). Intrinsics are scaled when depth resolution
differs from the reference 1280x720.
"""

import numpy as np

try:
    import open3d as o3d
except ImportError:
    o3d = None


class PointCloudConverter:
    """
    Converts depth images (mm) to point clouds (m) using Open3D's
    create_from_depth_image and pinhole model (same as pc_offiline.py).
    Intrinsics are for the rectified left ZED camera at 1280x720.
    """

    def __init__(self):
        # ZED intrinsics at reference resolution 1280x720 (from calibrator.K_zed)
        self._K = np.array(
            [
                [521.581604, 0.0, 636.33398438],
                [0.0, 521.581604, 373.10964966],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self._ref_w = 1280
        self._ref_h = 720

    def scale_intrinsics(self, width: int, height: int) -> np.ndarray:
        """
        Scale intrinsics to a different image size (e.g. when depth is not 1280x720).
        Same scaling as pc_offiline (fx, fy, cx, cy scaled by width/ref_w and height/ref_h).

        Parameters:
            width: Image width (pixels).
            height: Image height (pixels).

        Returns:
            K: 3x3 intrinsic matrix for the given resolution, dtype float64.
        """
        sx = width / self._ref_w
        sy = height / self._ref_h
        K_scaled = np.array(
            [
                [self._K[0, 0] * sx, 0.0, self._K[0, 2] * sx],
                [0.0, self._K[1, 1] * sy, self._K[1, 2] * sy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return K_scaled

    def depth_pixel_to_point(self, u: float, v: float, depth_m: float, width: int, height: int) -> np.ndarray:
        """
        Back-project one depth pixel into metric camera coordinates.

        Returns [x_right, y_down, z_forward] in meters.
        """
        K = self.scale_intrinsics(width, height)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        x = (float(u) - cx) * float(depth_m) / fx
        y = (float(v) - cy) * float(depth_m) / fy
        z = float(depth_m)
        return np.array([x, y, z], dtype=np.float32)

    def project_point_cloud_to_depth_image(
        self,
        points_xyz: np.ndarray,
        width: int,
        height: int,
        depth_scale: float = 1000.0,
        depth_max: float = 10.6,
    ) -> np.ndarray:
        """
        Project camera-centered points [x_right, y_up, z_forward] to a depth image.

        Returns a float32 depth image in meters on the requested image grid.
        """
        if o3d is None:
            raise ImportError("open3d is required for project_point_cloud_to_depth_image but is not installed")

        if points_xyz.size == 0:
            return np.zeros((height, width), dtype=np.float32)

        points_cam = np.asarray(points_xyz, dtype=np.float32).copy()
        points_cam[:, 1] *= -1.0  # Open3D pinhole projection uses image-space y-down.

        pcd = o3d.t.geometry.PointCloud()
        pcd.point["positions"] = o3d.core.Tensor(points_cam, dtype=o3d.core.Dtype.Float32)

        intrinsics = o3d.core.Tensor(self.scale_intrinsics(width, height).astype(np.float32))
        extrinsics = o3d.core.Tensor.eye(4, o3d.core.Dtype.Float32)
        depth = pcd.project_to_depth_image(
            width=width,
            height=height,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            depth_scale=float(depth_scale),
            depth_max=float(depth_max),
        )
        depth_np = depth.as_tensor().numpy()
        depth_np = np.asarray(depth_np).squeeze().astype(np.float32)
        if depth_np.size == 0:
            return np.zeros((height, width), dtype=np.float32)

        if np.nanmax(depth_np) > depth_max + 1.0:
            depth_np = depth_np / float(depth_scale)
        return depth_np.astype(np.float32)

    def depth_to_point_cloud(
        self,
        depth_mm: np.ndarray,
        depth_scale: float = 1000.0,
        max_depth_meters: float = 11.2,
    ) -> object:
        """
        Convert depth image to point cloud using Open3D's pinhole back-projection
        (same method as pc_offiline.depth_to_point_cloud_open3d). Depth in millimeters,
        output points in meters.

        Parameters:
            depth_mm: Depth image, shape (H, W), dtype float (millimeters).
                      Invalid depth (0, NaN, inf, or > depth_trunc) is skipped.

        Returns:
            Open3D PointCloud with points in meters. Uses the same
            depth_scale=depth_scale and depth_trunc=max_depth_meters.
        """
        if o3d is None:
            raise ImportError("open3d is required for depth_to_point_cloud but is not installed")

        h, w = depth_mm.shape[:2]
        K = self.scale_intrinsics(w, h)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=w,
            height=h,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
        )
        depth_o3d = o3d.geometry.Image(depth_mm.astype(np.float32))
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_o3d,
            o3d_intrinsic,  
            depth_scale=depth_scale,
            depth_trunc=max_depth_meters,
            project_valid_depth_only=True,
        )
        return pcd
