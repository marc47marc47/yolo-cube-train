#!/usr/bin/env python
"""Utility script to inspect CUDA and PyTorch availability."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Tuple


def run_nvidia_smi() -> Tuple[bool, str]:
    """Execute nvidia-smi and return (success flag, message)."""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        return False, "The command `nvidia-smi` is missing. Install CUDA Toolkit and add it to PATH."
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or str(exc)
        return False, f"`nvidia-smi` failed. Output: {detail}"
    else:
        preview = "\n".join(result.stdout.strip().splitlines()[:10])
        return True, preview


def main() -> int:
    print("=" * 60)
    print("CUDA & PyTorch environment check")
    print("=" * 60)
    print(f"Python: {Path(sys.executable).resolve()}")

    try:
        import torch  # type: ignore
    except ModuleNotFoundError:
        print("[!] PyTorch is not installed. Install it with:")
        print("    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
        return 1

    print(f"PyTorch version: {torch.__version__}")
    cuda_version = getattr(torch.version, "cuda", None)
    if not cuda_version or cuda_version == "0.0":
        print("[!] This PyTorch build does not include CUDA support.")
        print("    Install a CUDA-enabled build, for example:")
        print("    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
        success, message = run_nvidia_smi()
        if success:
            print("\n`nvidia-smi` output (first lines):")
            print(message)
        else:
            print(f"\n{message}")
        return 2

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"[+] Detected {gpu_count} CUDA GPU(s).")
        for idx in range(gpu_count):
            print(f"    GPU {idx}: {torch.cuda.get_device_name(idx)}")
        return 0

    print("[!] torch.cuda.is_available() = False")
    success, message = run_nvidia_smi()
    if success:
        print("`nvidia-smi` output (first lines):")
        print(message)
    else:
        print(message)
        print("\nSteps to install or repair CUDA:")
        print("  1. Download the correct installer from https://developer.nvidia.com/cuda-downloads.")
        print("  2. Reboot or restart the terminal after installation, then run `nvidia-smi` again.")
        print("  3. Check GPU drivers, BIOS settings, or virtualization if the problem persists.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
