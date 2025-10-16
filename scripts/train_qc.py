#!/usr/bin/env python
"""Train the quality control YOLO model with helpful environment checks."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Tuple

# Disable Ultralytics update prompts before the first import
os.environ.setdefault("ULTRALYTICS_NO_VERSION_CHECK", "1")
os.environ.setdefault("ULTRALYTICS_IGNORE_UPDATE", "1")
os.environ.setdefault("ULTRALYTICS_AUTO_UPDATE", "0")

try:
    from ultralytics import YOLO
except ModuleNotFoundError as exc:  # pragma: no cover - direct user guidance
    current_python = Path(sys.executable).resolve()
    message = (
        "\n[!] Cannot import 'ultralytics'. Make sure the project's virtual environment is active.\n"
        "    Recommended commands (PowerShell):\n"
        "      .\\yolo\\Scripts\\Activate.ps1\n"
        "      python scripts\\train_qc.py --data data\\quality_control\\dataset.yaml ...\n"
        "    Or run the script with the bundled interpreter:\n"
        "      .\\yolo\\Scripts\\python.exe scripts\\train_qc.py ...\n"
        "    If the package is missing, install it with: pip install ultralytics\n"
        f"    Current Python executable: {current_python}\n"
    )
    raise SystemExit(message) from exc


def _suppress_update_notice() -> None:
    """Monkey patch Ultralytics version checks to silence update logs."""
    try:
        from ultralytics.utils import checks as _checks  # type: ignore
    except Exception:
        return

    def _no_update() -> bool:
        return False

    for attr in ("check_pip_update_available", "check_latest_pypi_version"):
        if hasattr(_checks, attr):
            setattr(_checks, attr, _no_update)


_suppress_update_notice()


def _resolve_data_yaml(data_yaml: str) -> Path:
    """Ensure the dataset YAML exists and return an absolute path."""
    path = Path(data_yaml)
    if not path.exists():
        raise SystemExit(f"[!] Dataset YAML not found: {path.resolve()}")
    return path.resolve()


def _print_cuda_help() -> None:
    """Suggest CUDA troubleshooting steps and show nvidia-smi output."""
    print("    - Verify that the NVIDIA driver and CUDA Toolkit are installed.")
    print("    - Run `nvidia-smi` to confirm the driver status.")
    print("    - Download CUDA from https://developer.nvidia.com/cuda-downloads\n")
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        print("    - The `nvidia-smi` command is missing. Reinstall CUDA or add it to PATH.")
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or str(exc)
        print(f"    - `nvidia-smi` failed. Output: {detail}")
    else:
        preview = "\n".join(result.stdout.strip().splitlines()[:6])
        print("    `nvidia-smi` output (first lines):")
        print(preview)
        print()


def _normalize_device(device: str) -> Tuple[str, str]:
    """Validate the device argument and fall back to CPU when needed."""
    requested = (device or "").strip()
    if requested.lower() == "cpu":
        return "cpu", "CPU"

    try:
        import torch  # type: ignore
    except ModuleNotFoundError:
        print("[!] PyTorch is not installed. Falling back to CPU.")
        return "cpu", "CPU"

    cuda_version = getattr(torch.version, "cuda", None)
    if not cuda_version or cuda_version == "0.0":
        print("[!] The installed PyTorch build does not include CUDA. Training will use CPU.")
        print("    Install a CUDA-enabled build, for example:")
        print("    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
        _print_cuda_help()
        return "cpu", "CPU"

    if not torch.cuda.is_available():
        print("[!] CUDA is not available. Training will use CPU.")
        _print_cuda_help()
        return "cpu", "CPU"

    gpu_count = torch.cuda.device_count()
    if requested.lower() in {"", "auto", "cuda", "gpu"}:
        requested = "0"

    if requested.isdigit():
        index = int(requested)
        if index >= gpu_count:
            print(f"[!] Requested GPU index {index} exceeds available GPUs ({gpu_count}). Using GPU 0.")
            requested = "0"
    else:
        print(f"[!] Unrecognized device string '{requested}'. Using GPU 0.")
        requested = "0"

    gpu_name = torch.cuda.get_device_name(int(requested))
    return requested, gpu_name


def train_qc_model(
    data_yaml: str,
    model: str = "yolov8n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
    patience: int = 50,
    project: str = "artifacts/runs/qc",
    name: str = "exp",
):
    """
    訓練品質管理模型。

    Args:
        data_yaml: 資料集 YAML 檔案。
        model: 模型權重（可為檔案路徑或官方模型名稱）。
        epochs: 訓練輪數。
        imgsz: 輸入影像尺寸。
        batch: 批次大小。
        device: 運算裝置設定。
        patience: Early stopping patience。
        project: 專案輸出資料夾。
        name: 輸出子資料夾名稱。
    """
    data_yaml_path = _resolve_data_yaml(data_yaml)
    runtime_device, device_name = _normalize_device(device)

    model_path = Path(model)
    model_to_load = str(model_path.resolve()) if model_path.exists() else model

    print(f"\n{'='*60}")
    print("Starting quality control training")
    print(f"{'='*60}")
    print(f"Dataset     : {data_yaml_path}")
    print(f"Base model  : {model_to_load}")
    print(f"Epochs      : {epochs}")
    print(f"Image size  : {imgsz}")
    print(f"Batch size  : {batch}")
    if runtime_device == "cpu":
        print("Device      : CPU (GPU unavailable)")
    else:
        print(f"Device      : GPU {runtime_device} ({device_name})")
    print(f"{'='*60}\n")

    model_instance = YOLO(model_to_load)

    results = model_instance.train(
        data=str(data_yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=runtime_device,
        patience=patience,
        save=True,
        project=project,
        name=name,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        plots=True,
        val=True,
        save_period=-1,
    )

    print(f"\n{'='*60}")
    print("Training finished")
    print(f"{'='*60}")
    print(f"Best weights : {project}/{name}/weights/best.pt")
    print(f"Last weights : {project}/{name}/weights/last.pt")
    print(f"Results CSV  : {project}/{name}/results.csv")
    print(f"{'='*60}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="訓練品質管理模型")
    parser.add_argument("--data", required=True, help="資料集 YAML 檔案")
    parser.add_argument("--model", default="yolov8n.pt", help="模型路徑或官方模型名稱")
    parser.add_argument("--epochs", type=int, default=100, help="訓練輪數")
    parser.add_argument("--imgsz", type=int, default=640, help="影像尺寸")
    parser.add_argument("--batch", type=int, default=16, help="批次大小")
    parser.add_argument("--device", default="0", help="運算裝置 (GPU index 或 cpu)")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    parser.add_argument("--project", default="artifacts/runs/qc", help="輸出專案資料夾")
    parser.add_argument("--name", default="exp", help="訓練結果資料夾名稱")
    args = parser.parse_args()

    train_qc_model(
        data_yaml=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
