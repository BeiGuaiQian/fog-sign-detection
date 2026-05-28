"""Image quality and detection metric helpers."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pandas as pd
from skimage.measure import shannon_entropy


DETECTION_COLUMNS = [
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y",
    "width",
    "height",
    "area",
]


def _validate_bgr_image(bgr: np.ndarray, name: str = "bgr") -> None:
    """Validate an OpenCV BGR image."""
    if bgr is None:
        raise ValueError(f"{name} must not be None.")
    if not isinstance(bgr, np.ndarray):
        raise ValueError(f"{name} must be a NumPy array.")
    if bgr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"{name} must have shape (H, W, 3).")


def _round4(value: float | int) -> float:
    """Round a numeric value to four decimal places."""
    return round(float(value), 4)


def image_quality_metrics(bgr: np.ndarray) -> dict[str, float]:
    """Compute image quality metrics for a BGR image.

    Args:
        bgr: OpenCV-style BGR image.

    Returns:
        A dictionary containing brightness, contrast, entropy, sharpness, and
        edge statistics. Floating-point values are rounded to four decimals.

    Raises:
        ValueError: If the image is empty or not a three-channel BGR image.
    """
    _validate_bgr_image(bgr)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    edges = cv2.Canny(gray, 100, 200)
    edge_pixels = int(np.count_nonzero(edges))
    total_pixels = int(gray.shape[0] * gray.shape[1])

    return {
        "brightness_mean": _round4(np.mean(gray)),
        "contrast_std": _round4(np.std(gray)),
        "entropy": _round4(shannon_entropy(gray)),
        "laplacian_sharpness": _round4(laplacian.var()),
        "edge_pixels": edge_pixels,
        "edge_ratio": _round4(edge_pixels / total_pixels),
    }


def detection_summary(detections: list[dict[str, Any]]) -> dict[str, float | int]:
    """Summarize YOLO detection records.

    Args:
        detections: Detection dictionaries produced by ``detect_yolo.py``.

    Returns:
        A dictionary containing detection count, confidence statistics, and box
        area statistics. Empty detections produce zero-valued statistics.
    """
    if not detections:
        return {
            "detection_count": 0,
            "mean_confidence": 0,
            "max_confidence": 0,
            "total_area": 0,
            "mean_area": 0,
        }

    confidences = np.array(
        [float(detection.get("confidence", 0)) for detection in detections],
        dtype=np.float32,
    )
    areas = np.array(
        [float(detection.get("area", 0)) for detection in detections],
        dtype=np.float32,
    )

    return {
        "detection_count": int(len(detections)),
        "mean_confidence": _round4(np.mean(confidences)),
        "max_confidence": _round4(np.max(confidences)),
        "total_area": _round4(np.sum(areas)),
        "mean_area": _round4(np.mean(areas)),
    }


def compare_results(
    original_bgr: np.ndarray,
    dehazed_bgr: np.ndarray,
    original_dets: list[dict[str, Any]],
    dehazed_dets: list[dict[str, Any]],
) -> dict[str, float | int]:
    """Compare image quality and detection performance before and after dehazing.

    Args:
        original_bgr: Original foggy BGR image.
        dehazed_bgr: DCP-dehazed BGR image.
        original_dets: Detections from the original image.
        dehazed_dets: Detections from the dehazed image.

    Returns:
        A flat dictionary containing selected original/dehazed image metrics,
        detection metrics, and simple improvement values.

    Raises:
        ValueError: If either image is invalid.
    """
    original_quality = image_quality_metrics(original_bgr)
    dehazed_quality = image_quality_metrics(dehazed_bgr)
    original_detection = detection_summary(original_dets)
    dehazed_detection = detection_summary(dehazed_dets)

    confidence_improvement = (
        float(dehazed_detection["mean_confidence"])
        - float(original_detection["mean_confidence"])
    )
    detection_count_improvement = (
        int(dehazed_detection["detection_count"])
        - int(original_detection["detection_count"])
    )

    return {
        "original_brightness_mean": original_quality["brightness_mean"],
        "dehazed_brightness_mean": dehazed_quality["brightness_mean"],
        "original_contrast_std": original_quality["contrast_std"],
        "dehazed_contrast_std": dehazed_quality["contrast_std"],
        "original_entropy": original_quality["entropy"],
        "dehazed_entropy": dehazed_quality["entropy"],
        "original_laplacian_sharpness": original_quality["laplacian_sharpness"],
        "dehazed_laplacian_sharpness": dehazed_quality["laplacian_sharpness"],
        "original_edge_pixels": original_quality["edge_pixels"],
        "dehazed_edge_pixels": dehazed_quality["edge_pixels"],
        "original_detection_count": original_detection["detection_count"],
        "dehazed_detection_count": dehazed_detection["detection_count"],
        "original_mean_confidence": original_detection["mean_confidence"],
        "dehazed_mean_confidence": dehazed_detection["mean_confidence"],
        "confidence_improvement": _round4(confidence_improvement),
        "detection_count_improvement": detection_count_improvement,
    }


def detections_to_dataframe(detections: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert detection dictionaries to a table with fixed columns.

    Args:
        detections: Detection dictionaries produced by ``detect_yolo.py``.

    Returns:
        A Pandas DataFrame. If detections are empty, the DataFrame has the
        fixed detection columns and no rows.
    """
    if not detections:
        return pd.DataFrame(columns=DETECTION_COLUMNS)

    return pd.DataFrame(detections).reindex(columns=DETECTION_COLUMNS)


def compute_image_metrics(image: np.ndarray) -> dict[str, float]:
    """Compatibility wrapper for ``image_quality_metrics``.

    Args:
        image: OpenCV-style BGR image.

    Returns:
        A dictionary of image quality metric names and values.
    """
    return image_quality_metrics(image)


def build_metrics_table(original: np.ndarray, dehazed: np.ndarray) -> pd.DataFrame:
    """Build a simple original/dehazed image quality comparison table.

    Args:
        original: Original foggy BGR image.
        dehazed: Dehazed BGR image.

    Returns:
        A Pandas DataFrame containing image quality metrics for both images.
    """
    return pd.DataFrame(
        [
            {"image": "original", **image_quality_metrics(original)},
            {"image": "dehazed", **image_quality_metrics(dehazed)},
        ]
    )
