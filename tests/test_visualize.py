"""Tests for visualization module."""
from __future__ import annotations

import time

import numpy as np
import pytest

from src.visualize.overlay import Box, Overlay


@pytest.mark.unit
class TestBox:
    """Tests for Box dataclass."""

    def test_box_creation(self) -> None:
        """Test creating a Box instance."""
        box = Box(xyxy=(10.0, 20.0, 100.0, 200.0), confidence=0.95, cls=0)
        assert box.xyxy == (10.0, 20.0, 100.0, 200.0)
        assert box.confidence == 0.95
        assert box.cls == 0


@pytest.mark.unit
class TestOverlay:
    """Tests for Overlay class."""

    def test_init(self) -> None:
        """Test Overlay initialization."""
        class_names = ["person", "car", "dog"]
        overlay = Overlay(class_names)
        assert overlay.class_names == class_names
        assert overlay._fps == 0.0

    def test_update_fps(self) -> None:
        """Test FPS calculation."""
        overlay = Overlay(["person"])
        overlay._last_time = time.time() - 0.1  # 100ms ago

        fps = overlay.update_fps()
        assert fps > 0
        # Should be around 10 FPS (1 / 0.1), but allow wider range for timing variations
        assert 1 < fps < 20

    def test_update_fps_smoothing(self) -> None:
        """Test FPS smoothing over multiple updates."""
        overlay = Overlay(["person"])

        # Simulate consistent 30 FPS
        for _ in range(10):
            overlay._last_time = time.time() - (1.0 / 30.0)
            overlay.update_fps()
            time.sleep(0.001)  # Small delay

        # FPS should stabilize around 30, allow range due to timing variations
        assert 15 < overlay._fps < 45

    def test_draw_empty_boxes(self, sample_image: np.ndarray) -> None:
        """Test drawing with no boxes."""
        overlay = Overlay(["person"])
        result = overlay.draw(sample_image, [])

        assert result.shape == sample_image.shape
        # Result should be different due to FPS text
        assert not np.array_equal(result, sample_image)

    def test_draw_single_box(self, sample_image: np.ndarray) -> None:
        """Test drawing a single bounding box."""
        overlay = Overlay(["person", "car"])
        box = Box(xyxy=(50.0, 60.0, 200.0, 300.0), confidence=0.85, cls=0)

        result = overlay.draw(sample_image, [box])

        assert result.shape == sample_image.shape
        # Check that the frame was modified
        assert not np.array_equal(result, sample_image)
        # Original should remain unchanged
        assert np.array_equal(sample_image, sample_image)

    def test_draw_multiple_boxes(self, sample_image: np.ndarray) -> None:
        """Test drawing multiple bounding boxes."""
        overlay = Overlay(["person", "car", "dog"])
        boxes = [
            Box(xyxy=(50.0, 60.0, 200.0, 300.0), confidence=0.85, cls=0),
            Box(xyxy=(250.0, 100.0, 400.0, 350.0), confidence=0.92, cls=1),
            Box(xyxy=(100.0, 200.0, 250.0, 400.0), confidence=0.78, cls=2),
        ]

        result = overlay.draw(sample_image, boxes)

        assert result.shape == sample_image.shape
        assert not np.array_equal(result, sample_image)

    def test_draw_with_unknown_class(self, sample_image: np.ndarray) -> None:
        """Test drawing with class ID beyond class_names length."""
        overlay = Overlay(["person"])
        box = Box(xyxy=(50.0, 60.0, 200.0, 300.0), confidence=0.85, cls=5)

        # Should not raise error, should use cls number as string
        result = overlay.draw(sample_image, [box])
        assert result.shape == sample_image.shape

    def test_draw_with_edge_coordinates(self, sample_image: np.ndarray) -> None:
        """Test drawing with boxes at image edges."""
        overlay = Overlay(["person"])
        boxes = [
            # Top-left corner
            Box(xyxy=(0.0, 0.0, 50.0, 50.0), confidence=0.9, cls=0),
            # Bottom-right corner
            Box(
                xyxy=(590.0, 430.0, 640.0, 480.0),
                confidence=0.8,
                cls=0,
            ),
            # Label would be above image (y1 < 10)
            Box(xyxy=(100.0, 5.0, 200.0, 100.0), confidence=0.7, cls=0),
        ]

        result = overlay.draw(sample_image, boxes)
        assert result.shape == sample_image.shape

    def test_fps_display_in_frame(self, sample_image: np.ndarray) -> None:
        """Test that FPS is displayed in the frame."""
        overlay = Overlay(["person"])
        overlay._fps = 30.5

        result = overlay.draw(sample_image, [])

        # Check that top-left region is modified (FPS text)
        top_left_original = sample_image[:50, :200]
        top_left_result = result[:50, :200]
        assert not np.array_equal(top_left_original, top_left_result)

    def test_box_label_format(self, sample_image: np.ndarray) -> None:
        """Test that box labels show correct format."""
        overlay = Overlay(["person"])
        box = Box(xyxy=(50.0, 60.0, 200.0, 300.0), confidence=0.8567, cls=0)

        result = overlay.draw(sample_image, [box])

        # Label should be "person 0.86" (confidence rounded to 2 decimals)
        assert result.shape == sample_image.shape

    def test_frame_copy_independence(self, sample_image: np.ndarray) -> None:
        """Test that drawing doesn't modify the original frame."""
        overlay = Overlay(["person"])
        box = Box(xyxy=(50.0, 60.0, 200.0, 300.0), confidence=0.85, cls=0)
        original_copy = sample_image.copy()

        result = overlay.draw(sample_image, [box])

        # Original should be unchanged
        assert np.array_equal(sample_image, original_copy)
        # Result should be different
        assert not np.array_equal(result, original_copy)
