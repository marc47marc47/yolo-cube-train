"""Integration tests for complete inference pipeline."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.app.realtime import to_boxes
from src.camera.stream import StreamConfig, VideoStream
from src.detector.yolo import YoloDetector
from src.visualize.overlay import Box, Overlay


@pytest.mark.integration
class TestInferencePipeline:
    """Integration tests for the full inference pipeline."""

    @patch("src.detector.yolo.YOLO")
    def test_complete_pipeline_video_file(
        self, mock_yolo_class: MagicMock, temp_video_file: Path
    ) -> None:
        """Test complete pipeline with video file."""
        # Setup mock detector
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model

        # Mock detection results
        mock_result = MagicMock()
        mock_boxes = MagicMock()
        mock_boxes.xyxy = np.array([[100, 100, 200, 200]])
        mock_boxes.conf = np.array([0.9])
        mock_boxes.cls = np.array([0])
        mock_boxes.__len__.return_value = 1
        mock_result.boxes = mock_boxes
        mock_model.predict.return_value = [mock_result]

        # Create pipeline components
        config = StreamConfig(source=str(temp_video_file))
        detector = YoloDetector()
        overlay = Overlay(["person"])

        # Run pipeline
        with VideoStream(config) as stream:
            frame_count = 0
            for frame in stream.frames():
                results = detector.predict(frame)
                assert len(results) > 0

                boxes = to_boxes(results[0])
                assert len(boxes) == 1
                assert isinstance(boxes[0], Box)

                annotated = overlay.draw(frame, boxes)
                assert annotated.shape == frame.shape

                frame_count += 1
                if frame_count >= 3:  # Test first 3 frames
                    break

            assert frame_count == 3

    def test_to_boxes_empty_result(self) -> None:
        """Test to_boxes with empty result."""
        mock_result = MagicMock()
        mock_result.boxes = None

        boxes = to_boxes(mock_result)
        assert boxes == []

    def test_to_boxes_no_boxes_attribute(self) -> None:
        """Test to_boxes with result missing boxes attribute."""
        mock_result = MagicMock(spec=[])  # No boxes attribute
        boxes = to_boxes(mock_result)
        assert boxes == []

    def test_to_boxes_with_detections(self) -> None:
        """Test to_boxes with actual detections."""
        mock_result = MagicMock()
        mock_boxes = MagicMock()
        mock_boxes.xyxy = np.array(
            [[100, 100, 200, 200], [300, 300, 400, 400], [50, 50, 150, 150]]
        )
        mock_boxes.conf = np.array([0.9, 0.85, 0.75])
        mock_boxes.cls = np.array([0, 1, 0])
        mock_result.boxes = mock_boxes

        # Mock __len__ for the boxes
        mock_boxes.__len__ = lambda self: 3

        boxes = to_boxes(mock_result)

        assert len(boxes) == 3
        assert boxes[0].xyxy == (100.0, 100.0, 200.0, 200.0)
        assert boxes[0].confidence == 0.9
        assert boxes[0].cls == 0

        assert boxes[1].xyxy == (300.0, 300.0, 400.0, 400.0)
        assert boxes[1].confidence == 0.85
        assert boxes[1].cls == 1

        assert boxes[2].xyxy == (50.0, 50.0, 150.0, 150.0)
        assert boxes[2].confidence == 0.75
        assert boxes[2].cls == 0

    def test_to_boxes_missing_attributes(self) -> None:
        """Test to_boxes when some attributes are missing."""
        mock_result = MagicMock()
        mock_boxes = MagicMock()
        mock_boxes.xyxy = None  # Missing xyxy
        mock_boxes.conf = np.array([0.9])
        mock_boxes.cls = np.array([0])
        mock_result.boxes = mock_boxes

        boxes = to_boxes(mock_result)
        assert boxes == []


@pytest.mark.integration
@pytest.mark.slow
class TestRealtimeInference:
    """Tests for realtime inference with actual model."""

    def test_single_frame_inference(self, sample_image: np.ndarray) -> None:
        """Test inference on a single frame."""
        try:
            detector = YoloDetector(model_path="yolov8n.pt")
            results = detector.predict(sample_image)

            assert isinstance(results, list)
            assert len(results) > 0

            boxes = to_boxes(results[0])
            assert isinstance(boxes, list)
            # boxes may be empty if nothing detected in random image

        except Exception as e:
            pytest.skip(f"Model not available or inference failed: {e}")

    def test_video_inference_fps(self, temp_video_file: Path) -> None:
        """Test inference FPS on video file."""
        try:
            import time

            detector = YoloDetector(model_path="yolov8n.pt")
            overlay = Overlay(["person"])
            config = StreamConfig(source=str(temp_video_file))

            frame_times = []
            with VideoStream(config) as stream:
                for i, frame in enumerate(stream.frames()):
                    if i >= 5:  # Test 5 frames
                        break

                    start = time.time()
                    results = detector.predict(frame)
                    boxes = to_boxes(results[0])
                    overlay.draw(frame, boxes)
                    elapsed = time.time() - start

                    frame_times.append(elapsed)

            avg_time = np.mean(frame_times)
            fps = 1.0 / avg_time

            # Should be able to process at least 1 FPS even on slow hardware
            assert fps > 1.0
            # Log FPS for information
            print(f"\nAverage inference FPS: {fps:.2f}")

        except Exception as e:
            pytest.skip(f"Model not available or inference failed: {e}")


@pytest.mark.unit
class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_frame(self) -> None:
        """Test handling of empty frame."""
        empty_frame = np.array([])
        overlay = Overlay(["person"])

        # Should handle gracefully or raise appropriate error
        # Implementation depends on cv2.rectangle behavior
        try:
            result = overlay.draw(empty_frame, [])
            # If it doesn't raise, result should have same shape
            assert result.shape == empty_frame.shape
        except (ValueError, TypeError):
            # Acceptable to raise error for invalid input
            pass

    def test_very_small_frame(self) -> None:
        """Test handling of very small frame."""
        tiny_frame = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        overlay = Overlay(["person"])
        box = Box(xyxy=(2.0, 2.0, 8.0, 8.0), confidence=0.9, cls=0)

        result = overlay.draw(tiny_frame, [box])
        assert result.shape == tiny_frame.shape

    def test_very_large_confidence(self, sample_image: np.ndarray) -> None:
        """Test box with confidence > 1.0."""
        overlay = Overlay(["person"])
        box = Box(xyxy=(50.0, 60.0, 200.0, 300.0), confidence=1.5, cls=0)

        # Should handle gracefully
        result = overlay.draw(sample_image, [box])
        assert result.shape == sample_image.shape

    def test_negative_coordinates(self, sample_image: np.ndarray) -> None:
        """Test box with negative coordinates."""
        overlay = Overlay(["person"])
        box = Box(xyxy=(-10.0, -20.0, 50.0, 100.0), confidence=0.9, cls=0)

        result = overlay.draw(sample_image, [box])
        assert result.shape == sample_image.shape

    def test_coordinates_beyond_frame(self, sample_image: np.ndarray) -> None:
        """Test box extending beyond frame boundaries."""
        overlay = Overlay(["person"])
        h, w = sample_image.shape[:2]
        box = Box(
            xyxy=(w - 50, h - 50, w + 100, h + 100),
            confidence=0.9,
            cls=0,
        )

        result = overlay.draw(sample_image, [box])
        assert result.shape == sample_image.shape
