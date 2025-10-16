#!/usr/bin/env python
"""Verify pedestrian dataset structure and basic statistics."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Tuple

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate YOLO-style dataset paths and provide simple stats."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("data/pedestrian.yaml"),
        help="Path to YOLO dataset yaml file.",
    )
    return parser.parse_args()


def load_dataset_config(config_path: Path) -> Tuple[Path, Path, Path]:
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_path = (config_path.parent / cfg.get("path", ".")).resolve()
    train = (base_path / cfg["train"]).resolve()
    val = (base_path / cfg["val"]).resolve()
    test = (base_path / cfg.get("test", cfg["val"])).resolve()
    return train, val, test


def collect_pairs(image_dir: Path, label_dir: Path) -> Iterable[Tuple[Path, Path]]:
    extensions = (".jpg", ".jpeg", ".png", ".bmp")
    for image_path in image_dir.rglob("*"):
        if image_path.suffix.lower() not in extensions:
            continue
        label_path = label_dir / (image_path.stem + ".txt")
        yield image_path, label_path


def verify_split(split_name: str, split_root: Path) -> Counter:
    image_dir = split_root / "images"
    label_dir = split_root / "labels"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"[{split_name}] image directory missing: {image_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"[{split_name}] label directory missing: {label_dir}")

    stats = Counter()
    missing_labels = []
    empty_labels = []

    for image_path, label_path in collect_pairs(image_dir, label_dir):
        stats["images"] += 1
        if not label_path.is_file():
            missing_labels.append(label_path)
            continue
        with label_path.open("r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            empty_labels.append(label_path)
            continue

        stats["labeled_images"] += 1
        stats["annotations"] += len(lines)
        for line in lines:
            cls = line.split()[0]
            stats[f"class_{cls}"] += 1

    if missing_labels:
        stats["missing_labels"] = len(missing_labels)
    if empty_labels:
        stats["empty_labels"] = len(empty_labels)
    return stats


def main() -> int:
    args = parse_args()
    try:
        train_dir, val_dir, test_dir = load_dataset_config(args.config)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[verify_dataset] Failed to parse config: {exc}", file=sys.stderr)
        return 1

    print(f"[verify_dataset] Dataset config: {args.config.resolve()}")
    for name, root in (("train", train_dir.parent), ("val", val_dir.parent), ("test", test_dir.parent)):
        try:
            stats = verify_split(name, root)
        except FileNotFoundError as exc:
            print(f"[verify_dataset] {exc}", file=sys.stderr)
            return 1

        print(f"\n[{name}] root: {root}")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
