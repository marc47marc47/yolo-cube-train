"""Ultralytics YOLO detector wrapper."""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import numpy as np

try:
    from ultralytics import YOLO  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Ultralytics package not installed. Install via 'pip install ultralytics'."
    ) from exc


class YoloDetector:
    """Thin wrapper around Ultralytics YOLO model for inference."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        device: Optional[str] = None,
        conf: float = 0.25,
        iou: float = 0.7,
    ) -> None:
        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf
        self.iou = iou

    def predict(
        self,
        frame: np.ndarray,
        stream: bool = False,
        **kwargs: Any,
    ) -> List[Any]:
        """Run detection on a frame."""
        results = self.model.predict(
            frame,
            imgsz=kwargs.get("imgsz", 1280),  # Default to 1280 for higher resolution
            device=self.device,
            conf=self.conf,
            iou=self.iou,
            stream=stream,
            verbose=kwargs.get("verbose", False),
        )
        if stream:
            return list(results)
        return results

    @staticmethod
    def load_labels(path: Path) -> List[str]:
        """Load class names from yaml file."""
        import yaml

        with Path(path).open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        names = data.get("names")
        if isinstance(names, dict):
            return [names[k] for k in sorted(names)]
        if isinstance(names, list):
            return names
        raise ValueError(f"Unable to parse 'names' from {path}")
