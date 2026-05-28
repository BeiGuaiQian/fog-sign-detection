"""YOLOv8 detection utilities for traffic sign localization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_WEIGHTS_PATH = Path("weights") / "best.pt"
FALLBACK_MODEL_NAME = "yolov8n.pt"


def _project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[1]


def resolve_model_path(project_root: Path) -> tuple[Path | str, bool]:
    """Resolve the preferred YOLOv8 model path.

    Args:
        project_root: Root directory of the project.

    Returns:
        A tuple of ``(model_path_or_name, used_fallback)``. If
        ``weights/best.pt`` exists, the first item is a local Path. Otherwise it
        is the fallback model name ``yolov8n.pt``.
    """
    weights_path = project_root / DEFAULT_WEIGHTS_PATH
    if weights_path.exists():
        return weights_path, False
    return FALLBACK_MODEL_NAME, True


class TrafficSignDetector:
    """Traffic sign detector backed by Ultralytics YOLOv8."""

    def __init__(
        self,
        weight_path: str | Path = DEFAULT_WEIGHTS_PATH,
        device: str | None = None,
    ) -> None:
        """Load a YOLOv8 model for traffic sign detection.

        Args:
            weight_path: Path to custom weights. If this path does not exist,
                ``yolov8n.pt`` is loaded as a temporary fallback model.
            device: Inference device, such as ``"cpu"``, ``"0"``, or ``None``.

        Raises:
            ImportError: If the ``ultralytics`` package is not installed.
        """
        from ultralytics import YOLO

        self.device = device
        self.weight_path = Path(weight_path)
        if not self.weight_path.is_absolute():
            self.weight_path = _project_root() / self.weight_path

        self.used_fallback = not self.weight_path.exists()
        if self.used_fallback:
            print("未找到自定义权重，使用 yolov8n.pt 作为临时模型。")
            model_source: str | Path = FALLBACK_MODEL_NAME
        else:
            model_source = self.weight_path

        self.model = YOLO(str(model_source))
        self.names = self.model.names

    def detect(
        self,
        image_bgr: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.45,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Detect traffic signs in an OpenCV BGR image.

        Args:
            image_bgr: Input image in BGR channel order.
            conf: Confidence threshold passed to YOLOv8.
            iou: IoU threshold passed to YOLOv8.

        Returns:
            A tuple of ``(annotated_bgr, detections)``. ``annotated_bgr`` can be
            saved directly with ``cv2.imwrite``. For Streamlit display, convert
            it with ``cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)``.

        Raises:
            ValueError: If the input image or thresholds are invalid.
        """
        if image_bgr is None:
            raise ValueError("image_bgr must not be None.")
        if not isinstance(image_bgr, np.ndarray):
            raise ValueError("image_bgr must be a NumPy array.")
        if image_bgr.size == 0:
            raise ValueError("image_bgr must not be empty.")
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("image_bgr must have shape (H, W, 3).")
        if not 0 <= conf <= 1:
            raise ValueError("conf must be in the range [0, 1].")
        if not 0 <= iou <= 1:
            raise ValueError("iou must be in the range [0, 1].")

        results = self.model.predict(
            source=image_bgr,
            conf=conf,
            iou=iou,
            device=self.device,
            verbose=False,
        )

        if not results:
            return image_bgr.copy(), []

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return image_bgr.copy(), []

        detections = self._parse_detections(boxes)
        annotated_bgr = self._draw_detections(image_bgr, detections)

        return annotated_bgr, detections

    def detect_file(
        self,
        image_path: str | Path,
        conf: float = 0.25,
        iou: float = 0.45,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Read an image file and run traffic sign detection.

        Args:
            image_path: Path to an image readable by OpenCV.
            conf: Confidence threshold passed to YOLOv8.
            iou: IoU threshold passed to YOLOv8.

        Returns:
            A tuple of ``(annotated_bgr, detections)``.

        Raises:
            FileNotFoundError: If OpenCV cannot read the image.
        """
        path = Path(image_path)
        image_bgr = cv2.imread(str(path))
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        return self.detect(image_bgr, conf=conf, iou=iou)

    def _parse_detections(self, boxes: Any) -> list[dict[str, Any]]:
        """Convert YOLOv8 boxes into serializable detection dictionaries."""
        xyxy = boxes.xyxy.detach().cpu().numpy()
        cls_ids = boxes.cls.detach().cpu().numpy().astype(int)
        confs = boxes.conf.detach().cpu().numpy()

        detections: list[dict[str, Any]] = []
        for box, class_id, confidence in zip(xyxy, cls_ids, confs, strict=False):
            x1, y1, x2, y2 = [round(float(value), 1) for value in box]
            width = round(x2 - x1, 1)
            height = round(y2 - y1, 1)
            center_x = round((x1 + x2) / 2.0, 1)
            center_y = round((y1 + y2) / 2.0, 1)
            area = round(width * height, 1)
            class_name = self._class_name(class_id)

            detections.append(
                {
                    "class_id": int(class_id),
                    "class_name": class_name,
                    "confidence": round(float(confidence), 4),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "center_x": center_x,
                    "center_y": center_y,
                    "width": width,
                    "height": height,
                    "area": area,
                }
            )

        return detections

    def _draw_detections(
        self,
        image_bgr: np.ndarray,
        detections: list[dict[str, Any]],
    ) -> np.ndarray:
        """Draw detection boxes directly on a BGR image with OpenCV."""
        annotated = image_bgr.copy()
        for detection in detections:
            x1 = int(round(float(detection["x1"])))
            y1 = int(round(float(detection["y1"])))
            x2 = int(round(float(detection["x2"])))
            y2 = int(round(float(detection["y2"])))
            label = f'{detection["class_name"]} {detection["confidence"]:.2f}'

            color = (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            text_size, baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                1,
            )
            text_w, text_h = text_size
            label_y1 = max(0, y1 - text_h - baseline - 4)
            label_y2 = max(text_h + baseline + 4, y1)
            cv2.rectangle(
                annotated,
                (x1, label_y1),
                (x1 + text_w + 6, label_y2),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 3, label_y2 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        return annotated

    def _class_name(self, class_id: int) -> str:
        """Resolve a YOLO class id to a display name."""
        if isinstance(self.names, dict):
            return str(self.names.get(int(class_id), int(class_id)))
        if isinstance(self.names, list) and 0 <= int(class_id) < len(self.names):
            return str(self.names[int(class_id)])
        return str(class_id)


def detect_road_signs(image_path: Path, model_path: Path | str) -> list[dict[str, Any]]:
    """Detect road signs in an image with YOLOv8.

    Args:
        image_path: Path to the image file.
        model_path: Local weight path or Ultralytics model name.

    Returns:
        A list of detection records containing box coordinates, class labels,
        and confidence scores.
    """
    detector = TrafficSignDetector(weight_path=model_path)
    _, detections = detector.detect_file(image_path)
    return detections
