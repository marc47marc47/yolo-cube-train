"""Camera and video stream abstractions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Optional, Union

import cv2


@dataclass
class StreamConfig:
    """Configuration for a video stream."""

    source: Union[int, str] = 0
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[int] = None


class VideoStream:
    """Wrapper around cv2.VideoCapture with context manager support."""

    def __init__(self, config: StreamConfig) -> None:
        self._config = config
        self._capture: Optional[cv2.VideoCapture] = None

    def __enter__(self) -> "VideoStream":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def open(self) -> None:
        if self._capture and self._capture.isOpened():
            return
        self._capture = cv2.VideoCapture(self._config.source)
        if not self._capture.isOpened():
            raise RuntimeError(f"Unable to open video source: {self._config.source}")

        if self._config.width:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        if self._config.height:
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
        if self._config.fps:
            self._capture.set(cv2.CAP_PROP_FPS, self._config.fps)

    def release(self) -> None:
        if self._capture:
            self._capture.release()
            self._capture = None

    def fps(self) -> float:
        """Return the capture FPS if available, otherwise fallback to config or 30."""
        if not self._capture or not self._capture.isOpened():
            self.open()
        assert self._capture is not None
        fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = float(self._config.fps or 30.0)
        return fps

    def frame_size(self) -> tuple[int, int]:
        """Return (width, height) for the current capture."""
        if not self._capture or not self._capture.isOpened():
            self.open()
        assert self._capture is not None
        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width == 0 or height == 0:
            width = self._config.width or 640
            height = self._config.height or 480
        return width, height

    def frames(self) -> Generator[cv2.typing.MatLike, None, None]:
        """Yield frames until stream ends."""
        if not self._capture or not self._capture.isOpened():
            self.open()
        assert self._capture is not None  # mypy hint

        while True:
            ret, frame = self._capture.read()
            if not ret:
                break
            yield frame

    def read(self) -> Optional[cv2.typing.MatLike]:
        """Return a single frame or None if unavailable."""
        if not self._capture or not self._capture.isOpened():
            self.open()
        assert self._capture is not None
        ret, frame = self._capture.read()
        if not ret:
            return None
        return frame
