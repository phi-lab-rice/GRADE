"""
Weather Simulation Module (Functional API)

Implements fog and cloud effects using atmospheric scattering model.
Refactored to pure functions without a class wrapper.

Usage:
    from weather_cpu import add_fog, add_cloud

    # Fog: intensity 0-10
    foggy, mask = add_fog(img, intensity=5, return_attenuation=True)

    # Cloud: precise control
    cloudy, mask = add_cloud(img, num_clouds=5, cloud_size=0.3, intensity=8,
                           color="white", return_attenuation=True)
"""

import os
import random
import numpy as np
import cv2
from typing import Literal, Optional, Tuple, Union

try:
    from noise import pnoise3, pnoise2

    HAS_NOISE = True
except ImportError:
    HAS_NOISE = False
    print("Warning: 'noise' package not installed. Using random noise fallback.")


def _gen_perlin_noise(shape: Tuple[int, int], scale: float = 100.0) -> np.ndarray:
    """Generate Perlin noise for realistic fog density."""
    if not HAS_NOISE:
        noise = np.random.randn(*shape) * 50 + 128
        noise = cv2.GaussianBlur(noise.astype(np.float32), (31, 31), 0)
        return np.clip(noise, 0, 255)

    h, w = shape
    # Downsample for performance then resize
    d_h, d_w = min(h, 480), min(w, 480)

    noise = np.zeros((d_h, d_w), dtype=np.float32)
    s = 1.0 / scale

    # Simple 3-octave fractal noise
    for y in range(d_h):
        for x in range(d_w):
            v = pnoise2(x * s, y * s, octaves=4, persistence=0.5, lacunarity=2.0)
            noise[y, x] = (v + 1) * 128.0

    return cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)


def add_fog(
    image: np.ndarray,
    intensity: float = 5.0,  # 0 to 10
    return_attenuation: bool = False,
    cam_height: float = 20,
    fog_height: float = 100,
    haze_height: float = 35,
    airlight_color: Tuple[int, int, int] = (210, 210, 210),
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Add fog effect to image.

    Args:
        image: np.ndarray [H, W, 3] uint8
        intensity: float [0, 10], 0=No Fog, 10=Max Fog
        return_attenuation: bool, return (image, mask) if True

    Returns:
        Augmented Image (uint8) OR (Augmented Image, Attenuation Map [0,1])
    """
    if intensity <= 0:
        if return_attenuation:
            return image, np.ones(image.shape[:2], dtype=np.float32)
        return image

    h, w = image.shape[:2]
    image_float = image.astype(np.float32)

    # Map Intensity [0, 10] -> Visibility [High, Low] -> Beta [Low, High]
    # We map intensity directly to extinction coefficient (beta)
    # Intensity 10 -> visibility ~20m -> beta ~ 0.2
    # Intensity 1 -> visibility ~1000m -> beta ~ 0.004
    max_beta = 0.2  # at intensity 10
    beta = (intensity / 10.0) * max_beta

    # Airlight (fog color) - usually white/gray for fog
    airlight_color = np.asarray(airlight_color, dtype=np.float32)

    # 1. Height-based density
    elevation = np.ones((h, w), dtype=np.float32) * cam_height
    c = 1 - elevation / (fog_height + 1e-5)
    c = np.clip(c, 0, 1)

    # 2. Perlin Noise density variation
    noise = _gen_perlin_noise((h, w), scale=100.0 if w > 500 else 50.0)
    noise_factor = noise / 255.0

    # Combined Extinction Coefficient
    # beta_spatial = beta * c * noise_factor
    # Simplified: beta * noise helps create pockets
    # We keep the height term 'c' to make it ground-hugging if desired,
    # but for general camera views, uniform depth + noise is often enough.
    # Let's mix uniform and noisy for robust "intensity" feel.
    beta_spatial = beta * (0.6 + 0.4 * noise_factor)

    # 3. Distance Map (Scene Depth)
    # Without depth map, we assume a "corridor" or flat plane recession
    # y-coordinate approximation: bottom of image is close, top is far/sky
    # Normalized Y from 1.0 (bottom) to 4.0 (horizon/top)
    y_grad = np.linspace(4.0, 1.0, h).astype(np.float32)  # shape (h,)
    # specific distance model for visual effect
    depth_proxy = np.tile(y_grad[:, np.newaxis], (1, w))
    # Scale depth by fog height concept
    distance = depth_proxy * 10.0  # arbitrary scale meters

    # 4. Beer-Lambert Transmittance
    # T = exp(-beta * d)
    attenuation = np.exp(-beta_spatial * distance)

    # Apply
    # I = J * T + A * (1 - T)
    attenuation_3c = attenuation[:, :, np.newaxis]
    airlight_3c = np.ones_like(image_float) * airlight_color

    out = image_float * attenuation_3c + airlight_3c * (1 - attenuation_3c)
    out = np.clip(out, 0, 255).astype(np.uint8)

    if return_attenuation:
        return out, attenuation.astype(np.float32)
    return out


def add_cloud(
    image: np.ndarray,
    num_clouds: int = 4,
    cloud_size: Union[float, int] = 0.4,  # If float < 2, treated as ratio of min_dim
    intensity: float = 8.0,  # 0-10 opacity/density
    color: Literal["white", "gray"] = "white",
    return_attenuation: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Add synthetic clouds to image.

    Args:
        image: np.ndarray [H, W, 3]
        num_clouds: Number of cloud patches
        cloud_size: Size of clouds. If < 2.0, treated as ratio of image min dimension.
                    If > 2.0, treated as pixel size.
        intensity: [0, 10] cloud opacity/thickness
        color: "white" or "gray"
        return_attenuation: Return mask
    """
    if num_clouds <= 0 or intensity <= 0:
        if return_attenuation:
            return image, np.ones(image.shape[:2], dtype=np.float32)
        return image

    h, w = image.shape[:2]
    min_dim = min(h, w)
    image_float = image.astype(np.float32)

    # Determine absolute size
    if cloud_size < 2.0:
        base_size = int(min_dim * cloud_size)
    else:
        base_size = int(cloud_size)

    # Create Cloud Mask (Accumulator)
    # Starts at 0 (Clear/Transparent) -> 1 (Opaque/Cloudy)
    # We will invert to Transmittance at the end (1=Clear, 0=Blocked)
    cloud_density_map = np.zeros((h, w), dtype=np.float32)

    # Color
    if color == "white":
        cloud_rgb = np.array([210, 210, 210], dtype=np.float32)
    else:  # gray
        cloud_rgb = np.array([50, 50, 50], dtype=np.float32)

    # Pattern Directory (if available, else synthetic)
    pattern_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "AdverseWeatherSimulation",
        "data",
        "patterns",
    )
    has_patterns = os.path.exists(pattern_dir) and len(os.listdir(pattern_dir)) > 0

    for _ in range(num_clouds):
        # Random location
        cx = np.random.randint(0, w)
        cy = np.random.randint(
            0, h // 2
        )  # Clouds usually in sky/top half? Let's allow full image for "foggy cloud"
        # Actually user might want fog-like clouds anywhere. Let's do full range but bias top?
        # User asked for "add_cloud", usually implies sky, but for overlay tests fully random is safer.
        cy = np.random.randint(0, h)

        # Randomize size slightly
        this_size = int(base_size * random.uniform(0.8, 1.2))
        if this_size < 10:
            this_size = 10

        # Generate/Load Patch
        patch = None
        if has_patterns:
            try:
                fname = random.choice(
                    [f for f in os.listdir(pattern_dir) if f.endswith(("png", "jpg"))]
                )
                img_path = os.path.join(pattern_dir, fname)
                patch = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if patch is not None:
                    patch = (
                        cv2.resize(patch, (this_size, this_size)).astype(np.float32)
                        / 255.0
                    )
            except:
                pass

        if patch is None:
            # Synthetic Blob
            y, x = np.ogrid[
                -this_size // 2 : this_size // 2, -this_size // 2 : this_size // 2
            ]
            dist = np.sqrt(x * x + y * y)
            radius = this_size // 2
            patch = (1 - dist / radius).clip(0, 1)
            # Add noise
            noise = np.random.rand(*patch.shape) * 0.4
            patch = (patch + noise * patch).clip(0, 1)
            patch = cv2.GaussianBlur(patch, (15, 15), 0)

        # Paste Patch
        h_sh, w_sh = patch.shape
        x1 = cx - w_sh // 2
        y1 = cy - h_sh // 2
        x2 = x1 + w_sh
        y2 = y1 + h_sh

        # Crop to bounds
        pad_x1 = max(0, -x1)
        pad_y1 = max(0, -y1)
        crop_x1 = max(0, x1)
        crop_y1 = max(0, y1)
        crop_x2 = min(w, x2)
        crop_y2 = min(h, y2)

        patch_x1 = pad_x1
        patch_y1 = pad_y1
        patch_x2 = patch_x1 + (crop_x2 - crop_x1)
        patch_y2 = patch_y1 + (crop_y2 - crop_y1)

        if patch_x2 > patch.shape[1] or patch_y2 > patch.shape[0]:
            continue

        valid_patch = patch[patch_y1:patch_y2, patch_x1:patch_x2]

        # Accumulate density (max)
        cloud_density_map[crop_y1:crop_y2, crop_x1:crop_x2] = np.maximum(
            cloud_density_map[crop_y1:crop_y2, crop_x1:crop_x2], valid_patch
        )

    # Apply Intensity Scaling
    # Intensity [0, 10] -> Max Opacity [0, 1.0]
    max_opacity = min(1.0, intensity / 10.0)
    final_density = cloud_density_map * max_opacity

    # Transmittance = 1 - Density
    attenuation = 1.0 - final_density
    attenuation = np.clip(attenuation, 0, 1)

    # Composite
    attenuation_3c = attenuation[:, :, np.newaxis]
    cloud_color_layer = np.ones_like(image_float) * cloud_rgb

    # J * T + C * (1 - T)
    out = image_float * attenuation_3c + cloud_color_layer * (1 - attenuation_3c)
    out = np.clip(out, 0, 255).astype(np.uint8)

    if return_attenuation:
        return out, attenuation.astype(np.float32)
    return out
