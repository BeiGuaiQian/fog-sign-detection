"""Shared utility functions for paths, image I/O, and color conversion."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _validate_image(image: np.ndarray, name: str = "image") -> None:
    """Validate a non-empty NumPy image."""
    if image is None:
        raise ValueError(f"{name} must not be None.")
    if not isinstance(image, np.ndarray):
        raise ValueError(f"{name} must be a NumPy array.")
    if image.size == 0:
        raise ValueError(f"{name} must not be empty.")


def _validate_bgr_image(bgr: np.ndarray, name: str = "bgr") -> None:
    """Validate a non-empty OpenCV BGR image."""
    _validate_image(bgr, name)
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"{name} must have shape (H, W, 3).")


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist.

    Args:
        path: Directory path as a string or ``Path``.

    Returns:
        The created directory path as a ``Path`` object.
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to an OpenCV BGR image.

    Images with alpha channels are converted to RGB before channel reordering.

    Args:
        image: Input PIL image.

    Returns:
        BGR image as a NumPy array.

    Raises:
        ValueError: If ``image`` is not a valid PIL image.
    """
    if image is None:
        raise ValueError("image must not be None.")
    if not isinstance(image, Image.Image):
        raise ValueError("image must be a PIL Image.")

    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR image to RGB.

    Args:
        bgr: Input OpenCV-style BGR image.

    Returns:
        RGB image as a NumPy array.

    Raises:
        ValueError: If ``bgr`` is empty or not a three-channel image.
    """
    _validate_bgr_image(bgr)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_image_bgr(path: str | Path) -> np.ndarray:
    """Read an image from disk as an OpenCV BGR array.

    Args:
        path: Image path.

    Returns:
        BGR image loaded by OpenCV.

    Raises:
        FileNotFoundError: If OpenCV cannot read the image.
    """
    image_path = Path(path)
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    return bgr


def save_image(path: str | Path, bgr: np.ndarray) -> Path:
    """Save a BGR or single-channel image to disk.

    Args:
        path: Output image path.
        bgr: OpenCV-style BGR image or a single-channel grayscale image.

    Returns:
        The output path as a ``Path`` object.

    Raises:
        ValueError: If ``bgr`` is empty or has an unsupported shape.
        IOError: If OpenCV fails to write the image.
    """
    _validate_image(bgr, "bgr")
    if bgr.ndim == 3 and bgr.shape[2] == 3:
        image = bgr
    elif bgr.ndim == 2:
        image = bgr
    else:
        raise ValueError("bgr must have shape (H, W) or (H, W, 3).")

    output_path = Path(path)
    ensure_dir(output_path.parent)
    success = cv2.imwrite(str(output_path), image)
    if not success:
        raise OSError(f"Failed to save image: {output_path}")
    return output_path


def make_side_by_side(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    left_title: str = "Original",
    right_title: str = "Processed",
) -> np.ndarray:
    """Resize two BGR images to the same height and concatenate them.

    Titles are drawn in the upper-left corner of each image before stitching.

    Args:
        left_bgr: Left BGR image.
        right_bgr: Right BGR image.
        left_title: Title drawn on the left image.
        right_title: Title drawn on the right image.

    Returns:
        A side-by-side BGR comparison image.

    Raises:
        ValueError: If either image is empty or invalid.
    """
    _validate_bgr_image(left_bgr, "left_bgr")
    _validate_bgr_image(right_bgr, "right_bgr")

    target_height = min(left_bgr.shape[0], right_bgr.shape[0])
    left_resized = _resize_to_height(left_bgr, target_height)
    right_resized = _resize_to_height(right_bgr, target_height)

    left_labeled = _draw_title(left_resized, left_title)
    right_labeled = _draw_title(right_resized, right_title)
    return np.hstack([left_labeled, right_labeled])


def timestamp_name(prefix: str, suffix: str = ".jpg") -> str:
    """Create a timestamped file name.

    Args:
        prefix: File name prefix, such as ``"original"``.
        suffix: File extension, including or excluding the leading dot.

    Returns:
        A file name like ``original_20260528_153012.jpg``.
    """
    clean_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}{clean_suffix}"


def get_project_root() -> Path:
    """Return the project root directory.

    Returns:
        Absolute path of the project root.
    """
    return Path(__file__).resolve().parents[1]


def ensure_directory(path: str | Path) -> Path:
    """Compatibility wrapper for ``ensure_dir``."""
    return ensure_dir(path)


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to an RGB NumPy array.

    Args:
        image: Input PIL image.

    Returns:
        RGB image as a NumPy array.
    """
    if image is None:
        raise ValueError("image must not be None.")
    if not isinstance(image, Image.Image):
        raise ValueError("image must be a PIL Image.")
    return np.array(image.convert("RGB"))


def _resize_to_height(bgr: np.ndarray, target_height: int) -> np.ndarray:
    """Resize a BGR image while preserving aspect ratio."""
    height, width = bgr.shape[:2]
    if height == target_height:
        return bgr.copy()

    scale = target_height / height
    target_width = max(1, int(round(width * scale)))
    return cv2.resize(bgr, (target_width, target_height), interpolation=cv2.INTER_AREA)


def _draw_title(bgr: np.ndarray, title: str) -> np.ndarray:
    """Draw a readable title at the top-left of a BGR image."""
    labeled = bgr.copy()
    label = str(title)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    padding = 8
    text_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
    text_width, text_height = text_size

    x1, y1 = 0, 0
    x2 = min(labeled.shape[1], text_width + padding * 2)
    y2 = min(labeled.shape[0], text_height + baseline + padding * 2)

    overlay = labeled.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, labeled, 0.45, 0, labeled)
    cv2.putText(
        labeled,
        label,
        (padding, padding + text_height),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return labeled
