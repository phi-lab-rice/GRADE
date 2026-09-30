"""Canonical 3-D radar point-cloud extraction used by ``processor --pcd``."""

import numpy as np

from utils.extract_radar_data import iiqq_to_iq, mimo

RANGE_FFT_SIZE = 256
DOPPLER_FFT_SIZE = 64
AZIMUTH_FFT_SIZE = 32
ELEVATION_FFT_SIZE = 8
RANGE_RES_M = 0.04375
CAMERA_POS_IN_RADAR_M = np.array([-0.08, 0.10, 0.05], dtype=np.float32)


def _hann(length: int) -> np.ndarray:
    return np.ones(length, dtype=np.float32) if length <= 2 else np.hanning(length).astype(np.float32)


def _fft(data: np.ndarray, axis: int, size: int | None = None, shift: bool = False) -> np.ndarray:
    shape = [1] * data.ndim
    shape[axis] = data.shape[axis]
    result = np.fft.fft(data * _hann(data.shape[axis]).reshape(shape), n=size, axis=axis)
    return np.fft.fftshift(result, axes=axis) if shift else result


def _training(values: np.ndarray, index: int, win: int, guard: int, cyclic: bool):
    left = np.arange(index - guard - win, index - guard)
    right = np.arange(index + guard + 1, index + guard + win + 1)
    if cyclic:
        return values[left % len(values)], values[right % len(values)]
    if left[0] < 0 or right[-1] >= len(values):
        return None, None
    return values[left], values[right]


def _cfar(values: np.ndarray, axis: int, mode: int, win: int, guard: int, noise_div: int, cyclic: bool, threshold_db: float) -> np.ndarray:
    data = np.moveaxis(values, axis, -1)
    detected = np.zeros_like(data, dtype=bool)
    scale = 10.0 ** (threshold_db / 20.0)
    for outer in np.ndindex(data.shape[:-1]):
        row = data[outer]
        for cut in range(len(row)):
            left, right = _training(row, cut, win, guard, cyclic)
            if left is None:
                continue
            left_sum, right_sum = float(left.sum()), float(right.sum())
            if mode == 0:
                noise = (left_sum + right_sum) / (2**noise_div)
            elif mode == 1:
                noise = max(left_sum, right_sum) / (2**noise_div)
            else:
                noise = min(left_sum, right_sum) / (2**noise_div)
            detected[outer + (cut,)] = row[cut] > noise * scale
    return np.moveaxis(detected, -1, axis)


def _local_maxima(values: np.ndarray) -> np.ndarray:
    rows, cols = values.shape
    output = np.zeros_like(values, dtype=bool)
    for row in range(rows):
        for col in range(cols):
            center = values[row, col]
            maximum = True
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == dc == 0:
                        continue
                    rr = (row + dr) % rows
                    cc = col + dc
                    if cc < 0 or cc >= cols:
                        continue
                    if values[rr, cc] >= center:
                        maximum = False
                        break
                if not maximum:
                    break
            output[row, col] = maximum
    return output


def _xyz(snapshot: np.ndarray, range_m: float) -> np.ndarray | None:
    window = _hann(2)[:, None] * _hann(8)[None, :]
    spectrum = np.fft.fftshift(np.fft.fft2(snapshot * window, s=(ELEVATION_FFT_SIZE, AZIMUTH_FFT_SIZE)), axes=(0, 1))
    elevation, azimuth = np.unravel_index(np.abs(spectrum).argmax(), spectrum.shape)
    ux = float(np.clip(2.0 * (azimuth - AZIMUTH_FFT_SIZE // 2) / AZIMUTH_FFT_SIZE, -1, 1))
    uy = float(np.clip(2.0 * (elevation - ELEVATION_FFT_SIZE // 2) / ELEVATION_FFT_SIZE, -1, 1))
    uz_sq = 1.0 - ux * ux - uy * uy
    if uz_sq <= 0:
        return None
    return np.array([range_m * ux, range_m * uy, range_m * np.sqrt(uz_sq)], dtype=np.float32)


def _nearest_per_ray(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float32)
    ranges = np.linalg.norm(points, axis=1)
    valid = ranges > 1e-6
    points, ranges = points[valid], ranges[valid]
    chosen: dict[tuple[float, float, float], int] = {}
    for index, (direction, distance) in enumerate(zip(np.round(points / ranges[:, None], 6), ranges)):
        key = tuple(float(x) for x in direction)
        if key not in chosen or distance < ranges[chosen[key]]:
            chosen[key] = index
    return points[np.array(sorted(chosen.values()), dtype=np.int64)].astype(np.float32)


def extract_point_cloud_from_frame(iiqq_frame: np.ndarray) -> np.ndarray:
    """Return canonical ``(N, 3) float32`` camera-frame XYZ radar points."""
    virtual = mimo(iiqq_to_iq(iiqq_frame))
    range_fft = _fft(virtual, axis=-1, size=RANGE_FFT_SIZE)
    doppler_fft = _fft(range_fft, axis=0, size=DOPPLER_FFT_SIZE, shift=True)
    rd_map = np.sqrt(np.sum(np.abs(doppler_fft) ** 2, axis=(1, 2))).astype(np.float32)
    mask = _cfar(rd_map, 1, 2, 8, 4, 3, False, 15.0)
    mask &= _cfar(rd_map, 0, 0, 4, 2, 3, True, 15.0)
    mask &= _local_maxima(rd_map)
    points = []
    for doppler, range_bin in zip(*np.nonzero(mask)):
        range_m = float(range_bin) * RANGE_RES_M
        if range_m > 10.60:
            continue
        point = _xyz(doppler_fft[doppler, :, :, range_bin], range_m)
        if point is not None:
            points.append(point)
    if not points:
        return np.empty((0, 3), dtype=np.float32)
    return (_nearest_per_ray(np.stack(points, axis=0)) - CAMERA_POS_IN_RADAR_M).astype(np.float32)
