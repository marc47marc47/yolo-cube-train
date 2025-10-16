"""Visualization helpers for detection results."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import cv2
import numpy as np


@dataclass
class Box:
    xyxy: Tuple[float, float, float, float]
    confidence: float
    cls: int


class Overlay:
    """Draw bounding boxes, labels, and FPS on frames."""

    def __init__(self, class_names: List[str]) -> None:
        self.class_names = class_names
        self._last_time = time.time()
        self._fps = 0.0

    def update_fps(self) -> float:
        now = time.time()
        delta = now - self._last_time
        if delta > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / delta if self._fps else 1.0 / delta)
        self._last_time = now
        return self._fps

    def draw(self, frame: np.ndarray, boxes: Iterable[Box]) -> np.ndarray:
        frame_out = frame.copy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy)
            class_name = self.class_names[box.cls] if box.cls < len(self.class_names) else str(box.cls)
            color = (0, 255, 0)
            cv2.rectangle(frame_out, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {box.confidence:.2f}"
            cv2.putText(frame_out, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        fps = self.update_fps()
        cv2.putText(frame_out, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        return frame_out
