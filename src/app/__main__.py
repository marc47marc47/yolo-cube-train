"""Unified CLI entry point for YOLO project via `python -m src.app`."""
from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
