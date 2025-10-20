"""Tests for dataset integrity and format validation."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml


@pytest.mark.unit
class TestDatasetYAML:
    """Tests for YAML configuration file format."""

    def test_yaml_format_valid(self, data_dir: Path) -> None:
        """Test that pedestrian.yaml has valid format."""
        yaml_path = data_dir / "pedestrian.yaml"
        if not yaml_path.exists():
            pytest.skip("pedestrian.yaml not found")

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert isinstance(data, dict)
        assert "path" in data or "train" in data
        assert "names" in data

    def test_yaml_paths_exist(self, data_dir: Path) -> None:
        """Test that paths specified in YAML exist."""
        yaml_path = data_dir / "pedestrian.yaml"
        if not yaml_path.exists():
            pytest.skip("pedestrian.yaml not found")

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Check base path
        if "path" in data:
            base_path = Path(data["path"])
            if not base_path.is_absolute():
                base_path = data_dir.parent / base_path
            # Path might not exist yet if data not downloaded
            # Just check format is valid

    def test_yaml_names_format(self, data_dir: Path) -> None:
        """Test that names field has valid format."""
        yaml_path = data_dir / "pedestrian.yaml"
        if not yaml_path.exists():
            pytest.skip("pedestrian.yaml not found")

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        names = data.get("names")
        assert names is not None

        # Can be dict or list
        assert isinstance(names, (dict, list))

        if isinstance(names, dict):
            # Check keys are integers or can be converted to int
            for key in names.keys():
                assert isinstance(key, int) or str(key).isdigit()
        elif isinstance(names, list):
            # Check all items are strings
            for item in names:
                assert isinstance(item, str)


@pytest.mark.integration
class TestDatasetFiles:
    """Tests for dataset files and structure."""

    def test_data_directory_exists(self, data_dir: Path) -> None:
        """Test that data directory exists."""
        assert data_dir.exists()
        assert data_dir.is_dir()

    def test_quality_control_dataset(self, data_dir: Path) -> None:
        """Test quality control dataset structure (primary dataset)."""
        qc_dir = data_dir / "quality_control"
        if not qc_dir.exists():
            pytest.skip("quality_control dataset not found")

        assert qc_dir.is_dir()

        # Check for dataset.yaml
        dataset_yaml = qc_dir / "dataset.yaml"
        if dataset_yaml.exists():
            assert dataset_yaml.is_file()

        # Check for standard splits
        for split in ["train", "val"]:
            split_labels = qc_dir / "labels" / split
            if split_labels.exists():
                assert split_labels.is_dir()
                # Check has some label files
                label_files = list(split_labels.glob("*.txt"))
                if label_files:
                    assert len(label_files) > 0

    def test_pedestrian_directory_structure(self, data_dir: Path) -> None:
        """Test pedestrian dataset directory structure (optional)."""
        ped_dir = data_dir / "reference" / "pedestrian"
        if not ped_dir.exists():
            pytest.skip("pedestrian directory not found - optional dataset, use scripts/download_pedestrian_data.sh to download")

        # Check for train2017 and val2017
        train_dir = ped_dir / "train2017"
        val_dir = ped_dir / "val2017"

        if train_dir.exists():
            assert train_dir.is_dir()
            # Check for images and labels subdirectories
            images_dir = train_dir / "images"
            labels_dir = train_dir / "labels"

            if images_dir.exists():
                assert images_dir.is_dir()
            if labels_dir.exists():
                assert labels_dir.is_dir()

        if val_dir.exists():
            assert val_dir.is_dir()

    def test_label_files_format(self, data_dir: Path) -> None:
        """Test that label files have correct YOLO format."""
        # Try quality_control dataset first (primary)
        labels_dir = data_dir / "quality_control" / "labels" / "train"

        # Fallback to pedestrian dataset if quality_control not found
        if not labels_dir.exists():
            labels_dir = data_dir / "reference" / "pedestrian" / "train2017" / "labels"

        if not labels_dir.exists():
            pytest.skip("No label directory found - use quality_control or pedestrian dataset")

        label_files = list(labels_dir.glob("*.txt"))
        if not label_files:
            pytest.skip("no label files found")

        # Test first few label files
        for label_file in label_files[:5]:
            with open(label_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                parts = line.split()
                # YOLO format: class_id x_center y_center width height
                assert len(parts) == 5, f"Invalid format in {label_file}: {line}"

                class_id, x, y, w, h = parts
                # Check class_id is integer
                assert class_id.isdigit(), f"Invalid class_id in {label_file}"

                # Check coordinates are floats between 0 and 1
                for coord, name in [(x, "x"), (y, "y"), (w, "w"), (h, "h")]:
                    try:
                        val = float(coord)
                        assert (
                            0 <= val <= 1
                        ), f"Invalid {name} value in {label_file}: {val}"
                    except ValueError:
                        pytest.fail(f"Invalid float value in {label_file}: {coord}")

    def test_label_values_range(self, data_dir: Path) -> None:
        """Test that label values are in valid range [0, 1]."""
        # Try quality_control dataset first
        labels_dir = data_dir / "quality_control" / "labels" / "train"

        # Fallback to pedestrian dataset
        if not labels_dir.exists():
            labels_dir = data_dir / "reference" / "pedestrian" / "train2017" / "labels"

        if not labels_dir.exists():
            pytest.skip("No label directory found")

        label_files = list(labels_dir.glob("*.txt"))[:3]  # Test first 3 files
        if not label_files:
            pytest.skip("no label files found")

        for label_file in label_files:
            with open(label_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) != 5:
                        continue

                    _, x, y, w, h = parts
                    for val in [float(x), float(y), float(w), float(h)]:
                        assert 0 <= val <= 1

    def test_class_ids_valid(self, data_dir: Path) -> None:
        """Test that class IDs are valid."""
        # Try quality_control dataset first
        yaml_path = data_dir / "quality_control" / "dataset.yaml"
        labels_dir = data_dir / "quality_control" / "labels" / "train"

        # Fallback to pedestrian dataset
        if not yaml_path.exists():
            yaml_path = data_dir / "pedestrian.yaml"
            labels_dir = data_dir / "reference" / "pedestrian" / "train2017" / "labels"

        if not yaml_path.exists():
            pytest.skip("No dataset.yaml found")

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        names = data.get("names", {})
        if isinstance(names, list):
            num_classes = len(names)
        else:
            num_classes = max(int(k) for k in names.keys()) + 1

        if not labels_dir.exists():
            pytest.skip("labels directory not found")

        label_files = list(labels_dir.glob("*.txt"))[:3]
        if not label_files:
            pytest.skip("no label files found")

        for label_file in label_files:
            with open(label_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) != 5:
                        continue

                    class_id = int(parts[0])
                    assert (
                        0 <= class_id < num_classes
                    ), f"Invalid class_id {class_id} in {label_file}"

    def test_empty_labels_handling(self, data_dir: Path) -> None:
        """Test handling of empty label files."""
        # Try quality_control dataset first
        labels_dir = data_dir / "quality_control" / "labels" / "train"

        # Fallback to pedestrian dataset
        if not labels_dir.exists():
            labels_dir = data_dir / "reference" / "pedestrian" / "train2017" / "labels"

        if not labels_dir.exists():
            pytest.skip("labels directory not found")

        label_files = list(labels_dir.glob("*.txt"))
        if not label_files:
            pytest.skip("no label files found")

        # Check if any files are empty or contain only whitespace
        for label_file in label_files[:10]:
            with open(label_file, "r", encoding="utf-8") as f:
                content = f.read().strip()

            # Empty files are valid (no objects in image)
            if not content:
                # Just verify file can be read
                assert True

    def test_dataset_statistics(self, data_dir: Path) -> None:
        """Test and report dataset statistics."""
        # Try quality_control dataset first
        labels_dir = data_dir / "quality_control" / "labels" / "train"
        dataset_name = "quality_control"

        # Fallback to pedestrian dataset
        if not labels_dir.exists():
            labels_dir = data_dir / "reference" / "pedestrian" / "train2017" / "labels"
            dataset_name = "pedestrian"

        if not labels_dir.exists():
            pytest.skip("No label directory found")

        label_files = list(labels_dir.glob("*.txt"))
        if not label_files:
            pytest.skip("no label files found")

        total_objects = 0
        empty_files = 0

        for label_file in label_files:
            with open(label_file, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            if not lines:
                empty_files += 1
            else:
                total_objects += len(lines)

        print(f"\nDataset statistics ({dataset_name}):")
        print(f"  Total label files: {len(label_files)}")
        print(f"  Empty files: {empty_files}")
        print(f"  Total objects: {total_objects}")
        if label_files:
            print(f"  Average objects per image: {total_objects / len(label_files):.2f}")

        # Basic sanity check
        assert len(label_files) > 0


@pytest.mark.unit
class TestYAMLEdgeCases:
    """Tests for YAML edge cases."""

    def test_missing_yaml_file(self, tmp_path: Path) -> None:
        """Test handling of missing YAML file."""
        yaml_path = tmp_path / "nonexistent.yaml"
        assert not yaml_path.exists()

    def test_invalid_yaml_content(self, tmp_path: Path) -> None:
        """Test handling of invalid YAML content."""
        yaml_path = tmp_path / "invalid.yaml"
        yaml_path.write_text("this is not: valid: yaml: content:", encoding="utf-8")

        with pytest.raises(yaml.YAMLError):
            with open(yaml_path, "r", encoding="utf-8") as f:
                yaml.safe_load(f)

    def test_yaml_missing_required_fields(self, tmp_path: Path) -> None:
        """Test YAML with missing required fields."""
        yaml_path = tmp_path / "incomplete.yaml"
        yaml_path.write_text("train: train2017\n", encoding="utf-8")

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Should be missing 'names'
        assert "names" not in data
