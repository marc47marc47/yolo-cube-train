"""Unified CLI entry point for all application commands."""
from __future__ import annotations

import argparse
import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Main CLI entry point with subcommands."""
    parser = argparse.ArgumentParser(
        prog="python -m src.app",
        description="YOLO Project Unified CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Subcommands:
  realtime          Realtime detection application
  train-pedestrian  Train pedestrian detection model
  train-qc          Train quality control model
  eval              Evaluate model performance
  verify-data       Verify dataset structure
  prepare-data      Prepare training dataset
  analyze           Analyze quality labeled data
  inspect           Realtime quality inspection
  check-cuda        Check CUDA environment

Examples:
  python -m src.app realtime --source 0 --show
  python -m src.app train-qc --data data/qc.yaml --epochs 100
  python -m src.app check-cuda
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 1. Realtime detection
    parser_realtime = subparsers.add_parser("realtime", help="Realtime detection app")
    parser_realtime.add_argument("--source", default=0, help="Camera index or video/RTSP path")
    parser_realtime.add_argument("--model", default="yolov8n.pt", help="YOLO model weights path")
    parser_realtime.add_argument("--device", default=None, help="Device (cpu/0/1...)")
    parser_realtime.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser_realtime.add_argument("--iou", type=float, default=0.7, help="IoU threshold")
    parser_realtime.add_argument("--width", type=int, default=None, help="Force frame width")
    parser_realtime.add_argument("--height", type=int, default=None, help="Force frame height")
    parser_realtime.add_argument("--show", action="store_true", help="Show window")
    parser_realtime.add_argument("--names", default="data/pedestrian.yaml", help="Class names YAML file")
    parser_realtime.add_argument(
        "--mode", choices=["detect", "capture"], default="detect", help="Set to 'capture' to skip drawing boxes"
    )

    # 2. Train pedestrian
    parser_train_ped = subparsers.add_parser("train-pedestrian", help="Train pedestrian detection")
    parser_train_ped.add_argument("--config", default="data/pedestrian.yaml", help="Dataset config file")
    parser_train_ped.add_argument("--model", default="yolov8n.pt", help="Base model")
    parser_train_ped.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser_train_ped.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser_train_ped.add_argument("--batch", type=int, default=16, help="Batch size")
    parser_train_ped.add_argument("--device", default=None, help="Device")
    parser_train_ped.add_argument("--project", default="runs/detect", help="Project directory")
    parser_train_ped.add_argument("--name", default="train_pedestrian", help="Run name")
    parser_train_ped.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser_train_ped.add_argument("--save-latest", action="store_true", help="Save latest weights")

    # 3. Train QC
    parser_train_qc = subparsers.add_parser("train-qc", help="Train quality control model")
    parser_train_qc.add_argument("--data", required=True, help="Dataset YAML file")
    parser_train_qc.add_argument("--model", default="yolov8n.pt", help="Model path")
    parser_train_qc.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser_train_qc.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser_train_qc.add_argument("--batch", type=int, default=16, help="Batch size")
    parser_train_qc.add_argument("--device", default="0", help="Device")
    parser_train_qc.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    parser_train_qc.add_argument("--project", default="artifacts/runs/qc", help="Output directory")
    parser_train_qc.add_argument("--name", default="exp", help="Experiment name")

    # 4. Evaluate
    parser_eval = subparsers.add_parser("eval", help="Evaluate model performance")
    parser_eval.add_argument("--weights", default=None, help="Model weights path")
    parser_eval.add_argument("--config", default="data/pedestrian.yaml", help="Dataset config")
    parser_eval.add_argument("--device", default=None, help="Device")
    parser_eval.add_argument("--split", default="val", choices=["val", "test"], help="Eval split")

    # 5. Verify dataset
    parser_verify = subparsers.add_parser("verify-data", help="Verify dataset structure")
    parser_verify.add_argument("--config", default="data/pedestrian.yaml", help="Dataset config file")

    # 6. Prepare dataset
    parser_prepare = subparsers.add_parser("prepare-data", help="Prepare training dataset")
    parser_prepare.add_argument("--screenshots", default="artifacts/screenshots", help="Screenshots dir")
    parser_prepare.add_argument("--output", default="data/quality_control", help="Output dir")
    parser_prepare.add_argument(
        "--mode", default="binary", choices=["binary", "triclass", "quality"], help="Classification mode"
    )
    parser_prepare.add_argument("--train", type=float, default=0.7, help="Train ratio")
    parser_prepare.add_argument("--val", type=float, default=0.2, help="Val ratio")
    parser_prepare.add_argument("--test", type=float, default=0.1, help="Test ratio")

    # 7. Analyze
    parser_analyze = subparsers.add_parser("analyze", help="Analyze quality data")
    parser_analyze.add_argument("--dir", default="artifacts/screenshots", help="Screenshots dir")
    parser_analyze.add_argument("--export", action="store_true", help="Export quality list")
    parser_analyze.add_argument("--output", default="artifacts/quality_list.txt", help="Export file path")

    # 8. Quality Inspector
    parser_inspect = subparsers.add_parser("inspect", help="Realtime quality inspection")
    parser_inspect.add_argument("--model", required=True, help="Trained model path")
    parser_inspect.add_argument("--source", default="0", help="Video source")
    parser_inspect.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser_inspect.add_argument("--device", default=None, help="Device")

    # 9. Check CUDA
    subparsers.add_parser("check-cuda", help="Check CUDA environment")

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    # Route to appropriate handler
    if args.command == "realtime":
        from .realtime import main as realtime_main

        # Reconstruct args for realtime
        sys.argv = ["realtime"]
        for key, value in vars(args).items():
            if key != "command" and value is not None:
                if isinstance(value, bool):
                    if value:
                        sys.argv.append(f"--{key.replace('_', '-')}")
                else:
                    sys.argv.extend([f"--{key.replace('_', '-')}", str(value)])
        return realtime_main()

    elif args.command == "train-pedestrian":
        from src.training.train_pedestrian import main as train_ped_main

        sys.argv = ["train-pedestrian"]
        for key, value in vars(args).items():
            if key != "command" and value is not None:
                if isinstance(value, bool):
                    if value:
                        sys.argv.append(f"--{key.replace('_', '-')}")
                else:
                    sys.argv.extend([f"--{key.replace('_', '-')}", str(value)])
        return train_ped_main()

    elif args.command == "train-qc":
        from src.training.train_qc import main as train_qc_main

        sys.argv = ["train-qc"]
        for key, value in vars(args).items():
            if key != "command" and value is not None:
                sys.argv.extend([f"--{key.replace('_', '-')}", str(value)])
        return train_qc_main()

    elif args.command == "eval":
        from src.evaluation.evaluate_pedestrian import main as eval_main

        sys.argv = ["eval"]
        for key, value in vars(args).items():
            if key != "command" and value is not None:
                sys.argv.extend([f"--{key.replace('_', '-')}", str(value)])
        return eval_main()

    elif args.command == "verify-data":
        from src.data_utils.verify_dataset import main as verify_main

        sys.argv = ["verify-data", "--config", args.config]
        return verify_main()

    elif args.command == "prepare-data":
        from src.data_utils.prepare_quality_dataset import main as prepare_main

        sys.argv = ["prepare-data"]
        for key, value in vars(args).items():
            if key != "command":
                sys.argv.extend([f"--{key.replace('_', '-')}", str(value)])
        return prepare_main()

    elif args.command == "analyze":
        from src.analysis.analyze_quality_data import main as analyze_main

        sys.argv = ["analyze"]
        for key, value in vars(args).items():
            if key != "command":
                if isinstance(value, bool):
                    if value:
                        sys.argv.append(f"--{key.replace('_', '-')}")
                else:
                    sys.argv.extend([f"--{key.replace('_', '-')}", str(value)])
        return analyze_main()

    elif args.command == "inspect":
        from src.inference.quality_inspector import main as inspect_main

        sys.argv = ["inspect"]
        for key, value in vars(args).items():
            if key != "command":
                sys.argv.extend([f"--{key.replace('_', '-')}", str(value)])
        return inspect_main()

    elif args.command == "check-cuda":
        from src.utils.check_cuda import main as check_cuda_main

        return check_cuda_main()

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
