"""Realtime inference application tying together stream, detector, and overlay."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.camera.stream import StreamConfig, VideoStream
from src.detector.yolo import YoloDetector
from src.visualize.overlay import Box, Overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run realtime YOLO detection on camera/stream.")
    parser.add_argument("--source", default=0, help="Camera index or video/RTSP path.")
    parser.add_argument("--model", default="yolov8n.pt", help="Path to YOLO model weights.")
    parser.add_argument("--device", default=None, help="Device string for Ultralytics model (e.g. 'cpu' or '0').")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold.")
    parser.add_argument("--width", type=int, default=None, help="Force capture width.")
    parser.add_argument("--height", type=int, default=None, help="Force capture height.")
    parser.add_argument("--show", action="store_true", help="Display window; otherwise headless.")
    parser.add_argument(
        "--names",
        type=Path,
        default=Path("data/pedestrian.yaml"),
        help="YAML file defining class names (used for overlay).",
    )
    parser.add_argument(
        "--mode",
        choices=("detect", "capture"),
        default="detect",
        help="Select 'capture' to grab raw frames without drawing detection boxes.",
    )
    return parser.parse_args()


def to_boxes(result) -> list[Box]:  # type: ignore[no-untyped-def]
    boxes: list[Box] = []
    if not hasattr(result, "boxes"):
        return boxes
    ultralytics_boxes = result.boxes
    xyxy = getattr(ultralytics_boxes, "xyxy", None)
    conf = getattr(ultralytics_boxes, "conf", None)
    cls = getattr(ultralytics_boxes, "cls", None)
    if xyxy is None or conf is None or cls is None:
        return boxes

    total = len(ultralytics_boxes)
    for idx in range(total):
        boxes.append(
            Box(
                xyxy=tuple(map(float, xyxy[idx])),
                confidence=float(conf[idx]),
                cls=int(cls[idx]),
            )
        )
    return boxes


def main() -> int:
    args = parse_args()
    try:
        source: Optional[int | str] = int(args.source)
    except (ValueError, TypeError):
        source = args.source

    detection_enabled = args.mode == "detect"
    detector: Optional[YoloDetector] = None
    overlay: Optional[Overlay] = None

    if detection_enabled:
        detector = YoloDetector(model_path=args.model, device=args.device, conf=args.conf, iou=args.iou)
        class_names = YoloDetector.load_labels(args.names)
        overlay = Overlay(class_names)
    else:
        print("[realtime] Capture mode enabled (raw frames, no overlays).")
    stream_config = StreamConfig(source=source, width=args.width, height=args.height)

    stream = VideoStream(stream_config)
    try:
        stream.open()
    except RuntimeError as exc:
        if isinstance(source, int):
            print(f"[realtime] Camera index {source} not available.")
        else:
            print(f"[realtime] Unable to open video source '{source}'.")
        print(f"[realtime] Details: {exc}")
        return 1

    recording = False
    video_writer: Optional[cv2.VideoWriter] = None
    current_video_path: Optional[Path] = None
    recorded_frames = 0
    recorded_fps = 0.0

    def stop_recording() -> None:
        nonlocal recording, video_writer, current_video_path, recorded_frames
        if video_writer:
            video_writer.release()
            duration = (recorded_frames / recorded_fps) if recorded_fps else 0.0
            print(
                f"[record] saved {current_video_path} "
                f"({recorded_frames} frames, {duration:.1f}s)"
            )
        recording = False
        video_writer = None
        current_video_path = None

    with stream:

        def start_recording(frame: np.ndarray) -> None:
            nonlocal recording, video_writer, current_video_path, recorded_frames, recorded_fps
            videos_dir = Path("artifacts") / "videos"
            videos_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            current_video_path = videos_dir / f"capture_{timestamp}.mp4"

            height, width = frame.shape[:2]
            recorded_fps = stream.fps()
            if recorded_fps <= 0:
                recorded_fps = 30.0

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(current_video_path), fourcc, recorded_fps, (width, height))
            if not writer.isOpened():
                print(f"[record] failed to open video writer at {current_video_path}")
                current_video_path = None
                recorded_fps = 0.0
                return

            recording = True
            video_writer = writer
            recorded_frames = 0
            print(f"[record] started {current_video_path} (fps={recorded_fps:.2f})")

        try:
            for raw_frame in stream.frames():
                display_frame = raw_frame
                if detection_enabled and detector is not None and overlay is not None:
                    results = detector.predict(raw_frame, stream=False)
                    boxes = to_boxes(results[0]) if results else []
                    display_frame = overlay.draw(raw_frame, boxes)

                if recording and video_writer is not None:
                    video_writer.write(raw_frame)
                    recorded_frames += 1

                if args.show:
                    cv2.imshow("YOLO Realtime", display_frame)
                    key = cv2.waitKey(1) & 0xFF

                    if key in (ord("q"), 27):
                        break

                    if ord("0") <= key <= ord("9"):
                        quality = chr(key)
                        out_dir = Path("artifacts") / "screenshots" / f"quality_{quality}"
                        out_dir.mkdir(parents=True, exist_ok=True)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                        filename = out_dir / f"q{quality}_{timestamp}.jpg"
                        cv2.imwrite(str(filename), display_frame)
                        print(f"[capture] quality={quality} saved {filename}")

                    elif key == ord("s"):
                        out_dir = Path("artifacts") / "screenshots" / "unlabeled"
                        out_dir.mkdir(parents=True, exist_ok=True)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                        filename = out_dir / f"capture_{timestamp}.jpg"
                        cv2.imwrite(str(filename), display_frame)
                        print(f"[capture] unlabeled saved {filename}")

                    elif key == ord("v"):
                        if not recording:
                            start_recording(raw_frame)
                        else:
                            stop_recording()
                else:
                    # Headless mode: still allow recording if toggled externally in future.
                    pass
        finally:
            if recording:
                stop_recording()

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
