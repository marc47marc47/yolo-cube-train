"""Pytest configuration and shared fixtures."""
from __future__ import annotations

from pathlib import Path
from typing import Generator

import cv2
import numpy as np
import pytest


@pytest.fixture
def sample_image() -> np.ndarray:
    """Generate a simple test image (640x480 RGB)."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_gray_image() -> np.ndarray:
    """Generate a simple grayscale test image."""
    return np.random.randint(0, 255, (480, 640), dtype=np.uint8)


@pytest.fixture
def temp_video_file(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary test video file."""
    video_path = tmp_path / "test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))

    # Write 10 frames
    for _ in range(10):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        writer.write(frame)

    writer.release()
    yield video_path

    # Cleanup
    if video_path.exists():
        video_path.unlink()


@pytest.fixture
def project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def data_dir(project_root: Path) -> Path:
    """Get the data directory."""
    return project_root / "data"


@pytest.fixture
def artifacts_dir(project_root: Path) -> Path:
    """Get the artifacts directory."""
    return project_root / "artifacts"
