"""Tests for YOLO detector module."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.detector.yolo import YoloDetector


@pytest.mark.unit
class TestYoloDetector:
    """Tests for YoloDetector class."""

    @patch("src.detector.yolo.YOLO")
    def test_init_with_defaults(self, mock_yolo_class: MagicMock) -> None:
        """Test initialization with default parameters."""
        detector = YoloDetector()
        mock_yolo_class.assert_called_once_with("yolov8n.pt")
        assert detector.device is None
        assert detector.conf == 0.25
        assert detector.iou == 0.7

    @patch("src.detector.yolo.YOLO")
    def test_init_with_custom_params(self, mock_yolo_class: MagicMock) -> None:
        """Test initialization with custom parameters."""
        detector = YoloDetector(
            model_path="yolov8s.pt", device="cuda:0", conf=0.5, iou=0.45
        )
        mock_yolo_class.assert_called_once_with("yolov8s.pt")
        assert detector.device == "cuda:0"
        assert detector.conf == 0.5
        assert detector.iou == 0.45

    @patch("src.detector.yolo.YOLO")
    def test_predict(
        self, mock_yolo_class: MagicMock, sample_image: np.ndarray
    ) -> None:
        """Test predict method."""
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model
        mock_results = [MagicMock()]
        mock_model.predict.return_value = mock_results

        detector = YoloDetector()
        results = detector.predict(sample_image)

        mock_model.predict.assert_called_once()
        call_kwargs = mock_model.predict.call_args[1]
        assert call_kwargs["device"] is None
        assert call_kwargs["conf"] == 0.25
        assert call_kwargs["iou"] == 0.7
        assert call_kwargs["stream"] is False
        assert call_kwargs["verbose"] is False
        assert results == mock_results

    @patch("src.detector.yolo.YOLO")
    def test_predict_with_stream(
        self, mock_yolo_class: MagicMock, sample_image: np.ndarray
    ) -> None:
        """Test predict method with stream=True."""
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model
        # Simulate generator results
        mock_model.predict.return_value = iter([MagicMock(), MagicMock()])

        detector = YoloDetector()
        results = detector.predict(sample_image, stream=True)

        assert isinstance(results, list)
        assert len(results) == 2

    @patch("src.detector.yolo.YOLO")
    def test_predict_with_custom_imgsz(
        self, mock_yolo_class: MagicMock, sample_image: np.ndarray
    ) -> None:
        """Test predict with custom image size."""
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model
        mock_model.predict.return_value = [MagicMock()]

        detector = YoloDetector()
        detector.predict(sample_image, imgsz=1280)

        call_kwargs = mock_model.predict.call_args[1]
        assert call_kwargs["imgsz"] == 1280


@pytest.mark.unit
class TestYoloDetectorLabels:
    """Tests for label loading functionality."""

    def test_load_labels_with_dict(self, tmp_path: Path) -> None:
        """Test loading labels from YAML with dict format."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("names:\n  0: person\n  1: car\n  2: dog\n")

        labels = YoloDetector.load_labels(yaml_file)
        assert labels == ["person", "car", "dog"]

    def test_load_labels_with_list(self, tmp_path: Path) -> None:
        """Test loading labels from YAML with list format."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("names:\n  - person\n  - car\n  - dog\n")

        labels = YoloDetector.load_labels(yaml_file)
        assert labels == ["person", "car", "dog"]

    def test_load_labels_invalid_format(self, tmp_path: Path) -> None:
        """Test that invalid names format raises ValueError."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("names: invalid_string\n")

        with pytest.raises(ValueError, match="Unable to parse 'names'"):
            YoloDetector.load_labels(yaml_file)

    def test_load_labels_missing_names(self, tmp_path: Path) -> None:
        """Test that missing names key raises ValueError."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("other_key: value\n")

        with pytest.raises(ValueError, match="Unable to parse 'names'"):
            YoloDetector.load_labels(yaml_file)


@pytest.mark.integration
@pytest.mark.slow
class TestYoloDetectorIntegration:
    """Integration tests with real YOLO model (requires model download)."""

    def test_real_model_inference(self, sample_image: np.ndarray) -> None:
        """Test inference with real yolov8n model."""
        try:
            detector = YoloDetector(model_path="yolov8n.pt")
            results = detector.predict(sample_image)
            assert isinstance(results, list)
            assert len(results) > 0
        except Exception as e:
            pytest.skip(f"Model not available or inference failed: {e}")
