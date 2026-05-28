"""Command-line test tool for the YOLOv8 traffic sign detector."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.detect_yolo import TrafficSignDetector
from src.metrics import detections_to_dataframe
from src.utils import ensure_dir, save_image


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Test YOLOv8 detection on one image.")
    parser.add_argument(
        "--input",
        required=True,
        help="Input image path, for example data/samples/test.jpg.",
    )
    parser.add_argument(
        "--weights",
        default=str(Path("weights") / "best.pt"),
        help="YOLOv8 weight path. Default: weights/best.pt.",
    )
    parser.add_argument(
        "--output",
        default=str(Path("data") / "output"),
        help="Output directory. Default: data/output.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO confidence threshold. Default: 0.25.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="YOLO IoU threshold. Default: 0.45.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='Inference device, such as "cpu" or "0". Default: None.',
    )
    return parser.parse_args()


def resolve_project_path(path_text: str) -> Path:
    """Resolve a path relative to the project root when needed."""
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def print_detections(detections: list[dict]) -> None:
    """Print detection count and every detection record."""
    print(f"Detection count: {len(detections)}")
    if not detections:
        return

    for index, detection in enumerate(detections, start=1):
        print(
            f"[{index}] "
            f"class_id={detection['class_id']}, "
            f"class_name={detection['class_name']}, "
            f"confidence={detection['confidence']}, "
            f"box=({detection['x1']}, {detection['y1']}, "
            f"{detection['x2']}, {detection['y2']}), "
            f"center=({detection['center_x']}, {detection['center_y']}), "
            f"size=({detection['width']} x {detection['height']}), "
            f"area={detection['area']}"
        )


def main() -> int:
    """Run YOLOv8 detection and save annotated image plus CSV results."""
    args = parse_args()
    input_path = resolve_project_path(args.input)
    weights_path = resolve_project_path(args.weights)
    output_dir = ensure_dir(resolve_project_path(args.output))

    if not input_path.exists():
        print(f"Error: input image does not exist: {input_path}", file=sys.stderr)
        return 1

    try:
        detector = TrafficSignDetector(weight_path=weights_path, device=args.device)
        annotated_bgr, detections = detector.detect_file(
            input_path,
            conf=args.conf,
            iou=args.iou,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: YOLO detection failed: {exc}", file=sys.stderr)
        return 1

    annotated_path = save_image(output_dir / "annotated.jpg", annotated_bgr)
    detections_path = output_dir / "detections.csv"
    detections_to_dataframe(detections).to_csv(detections_path, index=False, encoding="utf-8-sig")

    print_detections(detections)
    print(f"Annotated image: {annotated_path}")
    print(f"Detections CSV: {detections_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
