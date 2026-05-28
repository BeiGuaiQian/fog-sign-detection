"""Command-line test tool for the DCP dehazing module."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dehaze_dcp import dehaze_dcp_bgr
from src.utils import ensure_dir, save_image


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Test DCP dehazing on one image.")
    parser.add_argument(
        "--input",
        required=True,
        help="Input foggy image path, for example data/samples/test.jpg.",
    )
    parser.add_argument(
        "--output",
        default=str(Path("data") / "output"),
        help="Output directory. Default: data/output.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=15,
        help="DCP dark channel window size. Default: 15.",
    )
    parser.add_argument(
        "--omega",
        type=float,
        default=0.95,
        help="DCP haze removal strength. Default: 0.95.",
    )
    parser.add_argument(
        "--t0",
        type=float,
        default=0.1,
        help="Minimum transmission threshold. Default: 0.1.",
    )
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


def main() -> int:
    """Run DCP dehazing and save result images."""
    args = parse_args()
    input_path = resolve_project_path(args.input)
    output_dir = ensure_dir(resolve_project_path(args.output))

    if not input_path.exists():
        print(f"Error: input image does not exist: {input_path}", file=sys.stderr)
        return 1

    bgr = cv2.imread(str(input_path))
    if bgr is None:
        print(f"Error: failed to read input image: {input_path}", file=sys.stderr)
        return 1

    try:
        dehazed_bgr, dark_channel_img, transmission_img, debug_info = dehaze_dcp_bgr(
            bgr,
            window_size=args.window_size,
            omega=args.omega,
            t0=args.t0,
            guided_radius=args.guided_radius,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    dehazed_path = save_image(output_dir / "dehazed.jpg", dehazed_bgr)
    dark_path = save_image(output_dir / "dark_channel.jpg", dark_channel_img)
    transmission_path = save_image(output_dir / "transmission.jpg", transmission_img)

    print(f"Atmospheric light: {debug_info['atmospheric_light']}")
    print(f"Dehazed image: {dehazed_path}")
    print(f"Dark channel: {dark_path}")
    print(f"Transmission map: {transmission_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
