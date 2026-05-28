"""Dark Channel Prior dehazing utilities.

This module implements the classical DCP pipeline:

1. dark channel estimation
2. atmospheric light estimation
3. transmission estimation
4. guided filtering refinement
5. scene radiance recovery

All public functions operate on NumPy arrays and do not depend on Streamlit, so
they can be reused by scripts, tests, and the web app.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _validate_float_image(image_float: np.ndarray, name: str = "image_float") -> None:
    """Validate a normalized three-channel floating-point image."""
    if image_float is None:
        raise ValueError(f"{name} must not be None.")
    if not isinstance(image_float, np.ndarray):
        raise ValueError(f"{name} must be a NumPy array.")
    if image_float.ndim != 3 or image_float.shape[2] != 3:
        raise ValueError(f"{name} must have shape (H, W, 3).")
    if image_float.size == 0:
        raise ValueError(f"{name} must not be empty.")


def _validate_window_size(window_size: int) -> int:
    """Validate and normalize the local filter window size."""
    if window_size <= 0:
        raise ValueError("window_size must be a positive integer.")
    if window_size % 2 == 0:
        window_size += 1
    return window_size


def _to_uint8_image(image_float: np.ndarray) -> np.ndarray:
    """Convert a normalized float image to uint8 in the 0-255 range."""
    return np.clip(image_float * 255.0, 0, 255).astype(np.uint8)


def dark_channel(image_float: np.ndarray, window_size: int = 15) -> np.ndarray:
    """Compute the dark channel of a normalized RGB or BGR image.

    The dark channel is obtained by first taking the minimum over the three
    color channels at each pixel, then applying a local minimum filter using
    ``cv2.erode``.

    Args:
        image_float: Input image with shape ``(H, W, 3)`` and values in ``0-1``.
        window_size: Size of the square local minimum filter window.

    Returns:
        A ``float32`` single-channel image with values clipped to ``0-1``.

    Raises:
        ValueError: If the input image or window size is invalid.
    """
    _validate_float_image(image_float)
    window_size = _validate_window_size(window_size)

    min_channel = np.min(image_float.astype(np.float32), axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dark = cv2.erode(min_channel, kernel)
    return np.clip(dark, 0.0, 1.0).astype(np.float32)


def estimate_atmospheric_light(
    image_float: np.ndarray,
    dark: np.ndarray,
    top_ratio: float = 0.001,
) -> np.ndarray:
    """Estimate atmospheric light from the brightest dark-channel pixels.

    Among the pixels with the largest dark-channel values, the pixel whose
    original three-channel intensity sum is largest is selected as atmospheric
    light ``A``.

    Args:
        image_float: Input image with shape ``(H, W, 3)`` and values in ``0-1``.
        dark: Dark channel image with shape ``(H, W)``.
        top_ratio: Ratio of highest dark-channel pixels used as candidates.

    Returns:
        Atmospheric light as a ``float32`` array with shape ``(3,)``.

    Raises:
        ValueError: If inputs are invalid.
    """
    _validate_float_image(image_float)
    if dark is None or not isinstance(dark, np.ndarray):
        raise ValueError("dark must be a NumPy array.")
    if dark.shape != image_float.shape[:2]:
        raise ValueError("dark must have shape (H, W), matching image_float.")
    if not 0 < top_ratio <= 1:
        raise ValueError("top_ratio must be in the range (0, 1].")

    flat_dark = dark.reshape(-1)
    flat_image = image_float.reshape(-1, 3)
    pixel_count = flat_dark.size
    candidate_count = max(1, int(pixel_count * top_ratio))

    candidate_indices = np.argpartition(flat_dark, -candidate_count)[-candidate_count:]
    candidate_pixels = flat_image[candidate_indices]
    brightness = np.sum(candidate_pixels, axis=1)
    best_index = candidate_indices[int(np.argmax(brightness))]

    return np.clip(flat_image[best_index], 0.0, 1.0).astype(np.float32)


def estimate_transmission(
    image_float: np.ndarray,
    atmospheric_light: np.ndarray,
    omega: float = 0.95,
    window_size: int = 15,
) -> np.ndarray:
    """Estimate the coarse transmission map using the DCP formula.

    The formula is ``t(x) = 1 - omega * dark_channel(I / A)``.

    Args:
        image_float: Input image with shape ``(H, W, 3)`` and values in ``0-1``.
        atmospheric_light: Atmospheric light array with shape ``(3,)``.
        omega: Haze removal strength. Commonly set to ``0.95``.
        window_size: Local dark channel window size.

    Returns:
        A ``float32`` transmission map with values clipped to ``0-1``.

    Raises:
        ValueError: If inputs are invalid.
    """
    _validate_float_image(image_float)
    if atmospheric_light is None or np.asarray(atmospheric_light).shape != (3,):
        raise ValueError("atmospheric_light must have shape (3,).")
    if not 0 <= omega <= 1:
        raise ValueError("omega must be in the range [0, 1].")

    safe_atmospheric_light = np.asarray(atmospheric_light, dtype=np.float32) + 1e-6
    normalized = image_float.astype(np.float32) / safe_atmospheric_light.reshape(1, 1, 3)
    raw_transmission = 1.0 - omega * dark_channel(normalized, window_size)
    return np.clip(raw_transmission, 0.0, 1.0).astype(np.float32)


def guided_filter(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int = 40,
    eps: float = 1e-3,
) -> np.ndarray:
    """Refine a single-channel image with guided filtering.

    Args:
        guide: Grayscale guide image with values in ``0-1``.
        src: Source image to filter, usually the coarse transmission map.
        radius: Radius of the square box filter window.
        eps: Regularization term used to stabilize local linear coefficients.

    Returns:
        The refined source image as ``float32``, clipped to ``0-1``.

    Raises:
        ValueError: If inputs are invalid.
    """
    if guide is None or src is None:
        raise ValueError("guide and src must not be None.")
    if not isinstance(guide, np.ndarray) or not isinstance(src, np.ndarray):
        raise ValueError("guide and src must be NumPy arrays.")
    if guide.ndim != 2 or src.ndim != 2:
        raise ValueError("guide and src must be single-channel images.")
    if guide.shape != src.shape:
        raise ValueError("guide and src must have the same shape.")
    if radius <= 0:
        raise ValueError("radius must be a positive integer.")
    if eps <= 0:
        raise ValueError("eps must be positive.")

    guide_float = guide.astype(np.float32)
    src_float = src.astype(np.float32)
    kernel_size = (radius * 2 + 1, radius * 2 + 1)

    mean_guide = cv2.boxFilter(guide_float, cv2.CV_32F, kernel_size, normalize=True)
    mean_src = cv2.boxFilter(src_float, cv2.CV_32F, kernel_size, normalize=True)
    mean_guide_src = cv2.boxFilter(
        guide_float * src_float,
        cv2.CV_32F,
        kernel_size,
        normalize=True,
    )
    cov_guide_src = mean_guide_src - mean_guide * mean_src

    mean_guide_sq = cv2.boxFilter(
        guide_float * guide_float,
        cv2.CV_32F,
        kernel_size,
        normalize=True,
    )
    var_guide = mean_guide_sq - mean_guide * mean_guide

    a = cov_guide_src / (var_guide + eps)
    b = mean_src - a * mean_guide

    mean_a = cv2.boxFilter(a, cv2.CV_32F, kernel_size, normalize=True)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, kernel_size, normalize=True)
    refined = mean_a * guide_float + mean_b

    return np.clip(refined, 0.0, 1.0).astype(np.float32)


def recover_scene_radiance(
    image_float: np.ndarray,
    atmospheric_light: np.ndarray,
    transmission: np.ndarray,
    t0: float = 0.1,
) -> np.ndarray:
    """Recover the haze-free scene radiance.

    The recovery formula is ``J(x) = (I(x) - A) / max(t(x), t0) + A``.

    Args:
        image_float: Input image with shape ``(H, W, 3)`` and values in ``0-1``.
        atmospheric_light: Atmospheric light array with shape ``(3,)``.
        transmission: Transmission map with shape ``(H, W)``.
        t0: Lower bound for transmission to avoid excessive amplification.

    Returns:
        Recovered image as ``float32`` with values clipped to ``0-1``.

    Raises:
        ValueError: If inputs are invalid.
    """
    _validate_float_image(image_float)
    if atmospheric_light is None or np.asarray(atmospheric_light).shape != (3,):
        raise ValueError("atmospheric_light must have shape (3,).")
    if transmission is None or transmission.shape != image_float.shape[:2]:
        raise ValueError("transmission must have shape (H, W), matching image_float.")
    if not 0 < t0 <= 1:
        raise ValueError("t0 must be in the range (0, 1].")

    atmospheric = np.asarray(atmospheric_light, dtype=np.float32).reshape(1, 1, 3)
    safe_transmission = np.maximum(transmission.astype(np.float32), t0)
    recovered = (image_float.astype(np.float32) - atmospheric) / safe_transmission[..., None]
    recovered = recovered + atmospheric
    return np.clip(recovered, 0.0, 1.0).astype(np.float32)


def dehaze_dcp_bgr(
    bgr: np.ndarray,
    window_size: int = 15,
    omega: float = 0.95,
    t0: float = 0.1,
    guided_radius: int = 40,
    guided_eps: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Dehaze a BGR image using Dark Channel Prior.

    Args:
        bgr: OpenCV-style BGR image with dtype ``uint8`` and values in ``0-255``.
        window_size: Local window size for dark channel and transmission.
        omega: Haze removal strength used in transmission estimation.
        t0: Minimum transmission used during radiance recovery.
        guided_radius: Radius for guided filtering.
        guided_eps: Regularization parameter for guided filtering.

    Returns:
        A tuple ``(dehazed_bgr, dark_channel_img, transmission_img, debug_info)``:
        ``dehazed_bgr`` is a uint8 BGR image, while ``dark_channel_img`` and
        ``transmission_img`` are uint8 single-channel images.

    Raises:
        ValueError: If the input image or parameters are invalid.
    """
    if bgr is None:
        raise ValueError("bgr must not be None.")
    if not isinstance(bgr, np.ndarray):
        raise ValueError("bgr must be a NumPy array.")
    if bgr.size == 0:
        raise ValueError("bgr must not be empty.")
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("bgr must have shape (H, W, 3).")

    window_size = _validate_window_size(window_size)
    if not 0 <= omega <= 1:
        raise ValueError("omega must be in the range [0, 1].")
    if not 0 < t0 <= 1:
        raise ValueError("t0 must be in the range (0, 1].")
    if guided_radius <= 0:
        raise ValueError("guided_radius must be a positive integer.")
    if guided_eps <= 0:
        raise ValueError("guided_eps must be positive.")

    image_float = bgr.astype(np.float32) / 255.0
    dark = dark_channel(image_float, window_size=window_size)
    atmospheric_light = estimate_atmospheric_light(image_float, dark)
    transmission = estimate_transmission(
        image_float,
        atmospheric_light,
        omega=omega,
        window_size=window_size,
    )

    guide_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    refined_transmission = guided_filter(
        guide_gray,
        transmission,
        radius=guided_radius,
        eps=guided_eps,
    )

    dehazed_float = recover_scene_radiance(
        image_float,
        atmospheric_light,
        refined_transmission,
        t0=t0,
    )

    dehazed_bgr = _to_uint8_image(dehazed_float)
    dark_channel_img = _to_uint8_image(dark)
    transmission_img = _to_uint8_image(refined_transmission)
    debug_info: dict[str, Any] = {
        "atmospheric_light": atmospheric_light.tolist(),
        "omega": omega,
        "t0": t0,
        "window_size": window_size,
        "guided_radius": guided_radius,
        "guided_eps": guided_eps,
        "image_shape": tuple(bgr.shape),
    }

    return dehazed_bgr, dark_channel_img, transmission_img, debug_info


def estimate_dark_channel(image: np.ndarray, window_size: int = 15) -> np.ndarray:
    """Compatibility wrapper for older code that used ``estimate_dark_channel``.

    Args:
        image: RGB or BGR image. ``uint8`` images are normalized automatically;
            floating-point images are assumed to already be in ``0-1``.
        window_size: Size of the square local minimum filter window.

    Returns:
        The dark channel as a ``float32`` image in ``0-1``.
    """
    image_float = image.astype(np.float32)
    if image_float.max(initial=0) > 1.0:
        image_float = image_float / 255.0
    return dark_channel(image_float, window_size=window_size)


def dehaze_image(image: np.ndarray) -> dict[str, np.ndarray]:
    """Compatibility wrapper that returns DCP outputs in a dictionary.

    Args:
        image: OpenCV-style BGR image.

    Returns:
        A dictionary containing ``dehazed_bgr``, ``dark_channel_img``,
        ``transmission_img``, and ``debug_info``.
    """
    dehazed_bgr, dark_channel_img, transmission_img, debug_info = dehaze_dcp_bgr(image)
    return {
        "dehazed_bgr": dehazed_bgr,
        "dark_channel_img": dark_channel_img,
        "transmission_img": transmission_img,
        "debug_info": debug_info,
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    sample_path = project_root / "data" / "samples" / "test.jpg"
    output_dir = project_root / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_bgr = cv2.imread(str(sample_path))
    if sample_bgr is None:
        raise ValueError(f"Sample image not found or unreadable: {sample_path}")

    result_bgr, result_dark, result_transmission, result_debug = dehaze_dcp_bgr(sample_bgr)
    cv2.imwrite(str(output_dir / "test_dehazed.jpg"), result_bgr)
    cv2.imwrite(str(output_dir / "test_dark_channel.jpg"), result_dark)
    cv2.imwrite(str(output_dir / "test_transmission.jpg"), result_transmission)
    print(result_debug)
