"""Tests for camera stream module."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.camera.stream import StreamConfig, VideoStream


@pytest.mark.unit
class TestStreamConfig:
    """Tests for StreamConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = StreamConfig()
        assert config.source == 0
        assert config.width is None
        assert config.height is None
        assert config.fps is None

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = StreamConfig(source="test.mp4", width=1920, height=1080, fps=30)
        assert config.source == "test.mp4"
        assert config.width == 1920
        assert config.height == 1080
        assert config.fps == 30


@pytest.mark.unit
class TestVideoStream:
    """Tests for VideoStream class."""

    def test_video_file_stream(self, temp_video_file: Path) -> None:
        """Test streaming from a video file."""
        config = StreamConfig(source=str(temp_video_file))
        with VideoStream(config) as stream:
            frame = stream.read()
            assert frame is not None
            assert frame.shape == (480, 640, 3)

    def test_video_file_frames_generator(self, temp_video_file: Path) -> None:
        """Test frames generator from video file."""
        config = StreamConfig(source=str(temp_video_file))
        with VideoStream(config) as stream:
            frames = list(stream.frames())
            assert len(frames) == 10
            for frame in frames:
                assert frame.shape == (480, 640, 3)

    def test_invalid_source_raises_error(self) -> None:
        """Test that invalid source raises RuntimeError."""
        config = StreamConfig(source="nonexistent.mp4")
        stream = VideoStream(config)
        with pytest.raises(RuntimeError, match="Unable to open video source"):
            stream.open()

    def test_context_manager(self, temp_video_file: Path) -> None:
        """Test context manager protocol."""
        config = StreamConfig(source=str(temp_video_file))
        with VideoStream(config) as stream:
            assert stream._capture is not None
            assert stream._capture.isOpened()
        # After exit, capture should be released
        assert stream._capture is None

    def test_read_without_open(self, temp_video_file: Path) -> None:
        """Test that read() opens stream automatically."""
        config = StreamConfig(source=str(temp_video_file))
        stream = VideoStream(config)
        frame = stream.read()
        assert frame is not None
        assert stream._capture is not None
        stream.release()

    def test_read_after_end_returns_none(self, temp_video_file: Path) -> None:
        """Test reading after video ends returns None."""
        config = StreamConfig(source=str(temp_video_file))
        with VideoStream(config) as stream:
            # Read all frames
            for _ in stream.frames():
                pass
            # Next read should return None
            frame = stream.read()
            assert frame is None

    def test_release_idempotent(self, temp_video_file: Path) -> None:
        """Test that calling release multiple times is safe."""
        config = StreamConfig(source=str(temp_video_file))
        stream = VideoStream(config)
        stream.open()
        stream.release()
        stream.release()  # Should not raise error
        assert stream._capture is None
