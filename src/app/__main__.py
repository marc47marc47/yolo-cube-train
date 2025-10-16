"""Entry point for running realtime YOLO app via `python -m src.app`."""
from __future__ import annotations

from .realtime import main

if __name__ == "__main__":
    raise SystemExit(main())
