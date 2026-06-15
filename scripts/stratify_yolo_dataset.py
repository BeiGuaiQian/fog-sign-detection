"""Create multilabel-stratified train/validation/test splits for a YOLO dataset."""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SOURCE_SPLITS = ("train", "valid", "val", "test")


@dataclass(frozen=True)
class Sample:
    """One paired YOLO image and label file."""

    image_path: Path
    label_path: Path
    class_ids: frozenset[int]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge and multilabel-stratify an existing YOLO dataset.",
    )
    parser.add_argument("--source", required=True, help="Source YOLO dataset directory.")
    parser.add_argument("--output", required=True, help="New dataset output directory.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output directory if it already exists.",
    )
    return parser.parse_args()


def load_config(source: Path) -> tuple[dict, list[str]]:
    """Load data.yaml and return its contents plus normalized class names."""
    config_path = source / "data.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing dataset config: {config_path}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw_names = config.get("names")
    if isinstance(raw_names, dict):
        names = [str(raw_names[index]) for index in range(len(raw_names))]
    elif isinstance(raw_names, list):
        names = [str(name) for name in raw_names]
    else:
        raise ValueError("data.yaml must define names as a list or integer-keyed mapping.")

    if config.get("nc", len(names)) != len(names):
        raise ValueError("data.yaml nc does not match the number of class names.")
    return config, names


def parse_label(label_path: Path, class_count: int) -> frozenset[int]:
    """Validate a YOLO label file and return the classes present in it."""
    class_ids: set[int] = set()
    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{label_path}:{line_number}: expected 5 columns.")

        class_id = int(parts[0])
        coords = [float(value) for value in parts[1:]]
        if not 0 <= class_id < class_count:
            raise ValueError(f"{label_path}:{line_number}: invalid class {class_id}.")
        if any(value < 0 or value > 1 for value in coords):
            raise ValueError(f"{label_path}:{line_number}: coordinates must be in [0, 1].")
        if coords[2] <= 0 or coords[3] <= 0:
            raise ValueError(f"{label_path}:{line_number}: box size must be positive.")
        class_ids.add(class_id)
    return frozenset(class_ids)


def collect_samples(source: Path, class_count: int) -> list[Sample]:
    """Collect all image/label pairs from existing YOLO split directories."""
    samples: list[Sample] = []
    seen_names: set[str] = set()

    for split in SOURCE_SPLITS:
        image_dir = source / split / "images"
        label_dir = source / split / "labels"
        if not image_dir.exists():
            continue
        if not label_dir.exists():
            raise FileNotFoundError(f"Missing labels directory: {label_dir}")

        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if image_path.name in seen_names:
                raise ValueError(f"Duplicate image filename across splits: {image_path.name}")

            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label for {image_path}: {label_path}")

            samples.append(
                Sample(
                    image_path=image_path,
                    label_path=label_path,
                    class_ids=parse_label(label_path, class_count),
                )
            )
            seen_names.add(image_path.name)

    if not samples:
        raise ValueError(f"No supported images found under {source}")
    return samples


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    """Validate split ratios."""
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(ratio <= 0 or ratio >= 1 for ratio in ratios):
        raise ValueError("All split ratios must be between 0 and 1.")
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("train, val, and test ratios must sum to 1.")


def split_samples(
    targets: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """Create multilabel-stratified train/validation/test indices."""
    indices = np.arange(len(targets))
    temporary_ratio = val_ratio + test_ratio
    first_splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=temporary_ratio,
        random_state=seed,
    )
    train_indices, temporary_indices = next(first_splitter.split(indices, targets))

    relative_test_ratio = test_ratio / temporary_ratio
    second_splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=relative_test_ratio,
        random_state=seed,
    )
    val_relative, test_relative = next(
        second_splitter.split(temporary_indices, targets[temporary_indices])
    )
    return {
        "train": train_indices,
        "valid": temporary_indices[val_relative],
        "test": temporary_indices[test_relative],
    }


def prepare_output(output: Path, overwrite: bool) -> None:
    """Create an empty output directory."""
    if output.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {output}. Use --overwrite to replace it."
            )
        shutil.rmtree(output)
    output.mkdir(parents=True)


def write_dataset(
    output: Path,
    samples: list[Sample],
    splits: dict[str, np.ndarray],
    names: list[str],
) -> None:
    """Copy split files and write data.yaml plus distribution statistics."""
    for split, indices in splits.items():
        image_dir = output / split / "images"
        label_dir = output / split / "labels"
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)

        for index in indices:
            sample = samples[int(index)]
            shutil.copy2(sample.image_path, image_dir / sample.image_path.name)
            shutil.copy2(sample.label_path, label_dir / sample.label_path.name)

    config = {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(names),
        "names": names,
    }
    (output / "data.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    rows: list[dict[str, str | int]] = []
    for class_id, class_name in enumerate(names):
        row: dict[str, str | int] = {
            "class_id": class_id,
            "class_name": class_name,
        }
        for split, indices in splits.items():
            row[split] = sum(
                class_id in samples[int(index)].class_ids for index in indices
            )
        rows.append(row)

    with (output / "class_distribution.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["class_id", "class_name", "train", "valid", "test"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    """Create and report the stratified dataset."""
    args = parse_args()
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    _, names = load_config(source)
    samples = collect_samples(source, len(names))
    targets = np.zeros((len(samples), len(names)), dtype=np.int8)
    for sample_index, sample in enumerate(samples):
        for class_id in sample.class_ids:
            targets[sample_index, class_id] = 1

    splits = split_samples(
        targets,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )
    prepare_output(output, args.overwrite)
    write_dataset(output, samples, splits, names)

    print(f"Source images: {len(samples)}")
    for split, indices in splits.items():
        print(f"{split}: {len(indices)} images")
    print(f"Output: {output}")
    print(f"Config: {output / 'data.yaml'}")
    print(f"Distribution: {output / 'class_distribution.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
