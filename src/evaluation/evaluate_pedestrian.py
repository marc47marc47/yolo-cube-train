#!/usr/bin/env python
"""Evaluate trained pedestrian model."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from ultralytics import YOLO  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLO pedestrian model.")
    parser.add_argument("--weights", type=Path, default=None, help="Path to model weights.")
    parser.add_argument("--config", type=Path, default=Path("data/pedestrian.yaml"), help="Dataset yaml path.")
    parser.add_argument("--device", type=str, default=None, help="Inference device.")
    parser.add_argument("--split", type=str, default="val", choices=("val", "test"), help="Dataset split to evaluate.")
    return parser.parse_args()


def resolve_weights(provided: Path | None) -> Path:
    if provided:
        return provided
    artifact = Path("artifacts/latest_model.txt")
    if not artifact.is_file():
        raise FileNotFoundError("Weights not specified and artifacts/latest_model.txt missing.")
    weight_path = Path(artifact.read_text(encoding="utf-8").strip())
    if not weight_path.is_file():
        raise FileNotFoundError(f"Weight file referenced in {artifact} not found: {weight_path}")
    return weight_path


def summarize_metrics(metrics: Dict[str, Any]) -> None:
    print("\n[metrics]")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def main() -> int:
    args = parse_args()
    weights = resolve_weights(args.weights)
    print(f"[evaluate_pedestrian] Evaluating weights: {weights}")
    model = YOLO(str(weights))
    metrics = model.val(data=str(args.config), split=args.split, device=args.device)
    if hasattr(metrics, "results_dict"):
        summarize_metrics(metrics.results_dict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
