"""Run batch dehazing and detection experiments for report analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dehaze_dcp import dehaze_dcp_bgr
from src.detect_yolo import TrafficSignDetector
from src.metrics import compare_results
from src.utils import ensure_dir, read_image_bgr, save_image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch compare original and DCP-dehazed traffic sign detection.",
    )
    parser.add_argument(
        "--input-dir",
        default=str(Path("data") / "samples"),
        help="Directory containing jpg, jpeg, and png images. Default: data/samples.",
    )
    parser.add_argument(
        "--weights",
        default=str(Path("weights") / "best.pt"),
        help="YOLOv8 weight path. Default: weights/best.pt.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path("data") / "output" / "batch"),
        help="Output directory. Default: data/output/batch.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO IoU threshold.")
    parser.add_argument(
        "--device",
        default=None,
        help='Inference device, such as "cpu" or "0". Default: None.',
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=15,
        help="DCP dark channel window size. Default: 15.",
    )
    parser.add_argument("--omega", type=float, default=0.95, help="DCP omega. Default: 0.95.")
    parser.add_argument("--t0", type=float, default=0.1, help="DCP t0. Default: 0.1.")
    parser.add_argument(
        "--guided-radius",
        type=int,
        default=40,
        help="Guided filter radius. Default: 40.",
    )
    return parser.parse_args()


def resolve_project_path(path_text: str) -> Path:
    """Resolve a path relative to the project root when needed."""
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def collect_images(input_dir: Path) -> list[Path]:
    """Collect supported image files from a directory."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def prepare_output_dirs(output_dir: Path) -> dict[str, Path]:
    """Create batch output directories."""
    return {
        "root": ensure_dir(output_dir),
        "dehazed": ensure_dir(output_dir / "dehazed"),
        "detect_original": ensure_dir(output_dir / "detect_original"),
        "detect_dehazed": ensure_dir(output_dir / "detect_dehazed"),
    }


def process_image(
    image_path: Path,
    detector: TrafficSignDetector,
    output_dirs: dict[str, Path],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Process one image and return a summary row."""
    row: dict[str, Any] = {
        "file_name": image_path.name,
        "file_path": str(image_path),
        "error": "",
    }

    try:
        original_bgr = read_image_bgr(image_path)
        dehazed_bgr, _, _, debug_info = dehaze_dcp_bgr(
            original_bgr,
            window_size=args.window_size,
            omega=args.omega,
            t0=args.t0,
            guided_radius=args.guided_radius,
        )
        original_detected_bgr, original_dets = detector.detect(
            original_bgr,
            conf=args.conf,
            iou=args.iou,
        )
        dehazed_detected_bgr, dehazed_dets = detector.detect(
            dehazed_bgr,
            conf=args.conf,
            iou=args.iou,
        )

        stem = image_path.stem
        save_image(output_dirs["dehazed"] / f"{stem}_dehazed.jpg", dehazed_bgr)
        save_image(
            output_dirs["detect_original"] / f"{stem}_detect_original.jpg",
            original_detected_bgr,
        )
        save_image(
            output_dirs["detect_dehazed"] / f"{stem}_detect_dehazed.jpg",
            dehazed_detected_bgr,
        )

        row.update(compare_results(original_bgr, dehazed_bgr, original_dets, dehazed_dets))
        row["atmospheric_light"] = debug_info["atmospheric_light"]
    except Exception as exc:
        row["error"] = str(exc)

    return row


def save_summaries(rows: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    """Save per-image summary and mean numeric summary CSV files."""
    summary_df = pd.DataFrame(rows)
    summary_path = output_dir / "summary.csv"
    mean_path = output_dir / "summary_mean.csv"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    numeric_df = summary_df.select_dtypes(include="number")
    if numeric_df.empty:
        mean_df = pd.DataFrame()
    else:
        mean_df = numeric_df.mean(numeric_only=True).to_frame(name="mean").reset_index()
        mean_df = mean_df.rename(columns={"index": "metric"})
    mean_df.to_csv(mean_path, index=False, encoding="utf-8-sig")

    return summary_path, mean_path


def main() -> int:
    """Run the batch experiment."""
    args = parse_args()
    input_dir = resolve_project_path(args.input_dir)
    weights_path = resolve_project_path(args.weights)
    output_dir = resolve_project_path(args.output_dir)

    try:
        image_paths = collect_images(input_dir)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not image_paths:
        print(f"Error: no jpg, jpeg, or png images found in: {input_dir}", file=sys.stderr)
        return 1

    output_dirs = prepare_output_dirs(output_dir)

    try:
        detector = TrafficSignDetector(weight_path=weights_path, device=args.device)
    except Exception as exc:
        print(f"Error: failed to load YOLO detector: {exc}", file=sys.stderr)
        return 1

    rows: list[dict[str, Any]] = []
    total = len(image_paths)
    for index, image_path in enumerate(image_paths, start=1):
        row = process_image(image_path, detector, output_dirs, args)
        rows.append(row)

        original_count = row.get("original_detection_count", 0)
        dehazed_count = row.get("dehazed_detection_count", 0)
        if row.get("error"):
            print(f"[{index}/{total}] {image_path.name} failed: {row['error']}")
        else:
            print(
                f"[{index}/{total}] {image_path.name} "
                f"original_dets={original_count}, dehazed_dets={dehazed_count}"
            )

    summary_path, mean_path = save_summaries(rows, output_dirs["root"])
    print(f"Summary CSV: {summary_path}")
    print(f"Summary mean CSV: {mean_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
