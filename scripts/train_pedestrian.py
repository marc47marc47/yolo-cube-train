#!/usr/bin/env python
"""Train YOLO model on pedestrian dataset using Ultralytics API."""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 on pedestrian dataset.")
    parser.add_argument("--config", type=Path, default=Path("data/pedestrian.yaml"), help="Dataset yaml path.")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model weights.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", type=str, default=None, help="Device for training (e.g. '0' or 'cpu').")
    parser.add_argument("--project", type=Path, default=Path("runs/detect"), help="Training project directory.")
    parser.add_argument("--name", type=str, default="train_pedestrian", help="Run name under project dir.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping epochs patience.")
    parser.add_argument("--save-latest", action="store_true", help="Save latest checkpoints.")
    return parser.parse_args()


def save_artifact_weights(run_dir: Path) -> None:
    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.is_file():
        print(f"[train_pedestrian] Best weights not found at {best_weights}")
        return
    artifact_dir = Path("artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_file = artifact_dir / "latest_model.txt"
    output_file.write_text(str(best_weights.resolve()), encoding="utf-8")
    print(f"[train_pedestrian] Recorded best weights at {output_file}")


def main() -> int:
    args = parse_args()
    model = YOLO(args.model)
    train_args = dict(
        data=str(args.config),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(args.project),
        name=args.name,
        patience=args.patience,
        save=args.save_latest,
        device=args.device,
    )
    print(f"[train_pedestrian] Training with args: {train_args}")
    results = model.train(**train_args)
    run_dir = Path(results.save_dir) if hasattr(results, "save_dir") else Path(args.project) / args.name
    save_artifact_weights(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
