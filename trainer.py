"""Trainer script for the smart flame detector dataset using Ultralytics YOLO."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = PROJECT_ROOT / "data.yaml"
DEFAULT_WEIGHTS = "yolov8s.pt"
DEFAULT_PROJECT_DIR = PROJECT_ROOT / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLO model on the smart flame detector dataset with GPU acceleration."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Path to the dataset YAML file (defaults to data.yaml in the repo).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=DEFAULT_WEIGHTS,
        help="Initial weights to fine-tune, e.g. yolov8s.pt.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="CUDA device identifier (e.g. 0, 1, '0,1'). Defaults to the first GPU.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=DEFAULT_PROJECT_DIR,
        help="Directory where Ultralytics will create the run folder.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="smart-flame-detector",
        help="Name of the training run subdirectory.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="auto",
        help="Optimizer to use (Ultralytics accepts auto, SGD, Adam, AdamW, etc.).",
    )
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Only set if you are okay running on CPU when CUDA is unavailable.",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Cache images for faster training (requires enough CPU RAM).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the most recent run for the selected project/name pair.",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Set if existing project/name directories should be reused.",
    )
    return parser.parse_args()


def configure_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("smart_flame_trainer")


def resolve_split_path(cfg_dir: Path, entry: str) -> Path:
    """Resolve dataset split paths even if the YAML contains misplaced ../ segments."""
    path = Path(entry)
    candidates = []

    if path.is_absolute():
        candidates.append(path)

    candidates.append((cfg_dir / path).resolve())

    stripped = str(path).lstrip("./\\")
    if stripped != str(path):
        candidates.append((cfg_dir / stripped).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not locate dataset split '{entry}' relative to '{cfg_dir}'.")


def materialize_data_config(data_cfg: Path, logger: logging.Logger) -> Path:
    """Write a resolved copy of data.yaml so Ultralytics always receives valid absolute paths."""
    data: dict[str, Any]
    with data_cfg.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    cfg_dir = data_cfg.parent
    for split in ("train", "val", "test"):
        if split in data and data[split]:
            resolved = resolve_split_path(cfg_dir, data[split])
            data[split] = str(resolved)
            logger.info("Resolved %s split to %s", split, resolved)

    resolved_path = data_cfg.with_name(f"{data_cfg.stem}.resolved.yaml")
    with resolved_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
    logger.info("Materialized dataset config at %s", resolved_path)
    return resolved_path


def choose_device(preferred: str | None, allow_cpu: bool) -> str:
    """Ensure training runs on the GPU unless CPU fallback is explicitly allowed."""
    if torch.cuda.is_available():
        return preferred if preferred is not None else "0"

    if allow_cpu:
        print(
            "[WARN] CUDA is not available; falling back to CPU because --allow-cpu-fallback was set.",
            file=sys.stderr,
        )
        return "cpu"

    raise RuntimeError(
        "CUDA device not detected. Install GPU drivers or rerun with --allow-cpu-fallback if CPU training is acceptable."
    )


def main() -> None:
    logger = configure_logging()
    args = parse_args()
    logger.info("Starting training run with args: %s", args)
    data_cfg = materialize_data_config(args.data, logger)
    device = choose_device(args.device, args.allow_cpu_fallback)
    logger.info("Using device: %s", device)

    model = YOLO(args.weights)
    logger.info("Loaded model weights from %s", args.weights)
    logger.info(
        "Beginning training for %s epochs | batch=%s | imgsz=%s",
        args.epochs,
        args.batch,
        args.imgsz,
    )
    results = model.train(
        data=str(data_cfg),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(args.project),
        name=args.name,
        optimizer=args.optimizer,
        cache=args.cache,
        resume=args.resume,
        exist_ok=args.exist_ok,
    )

    logger.info("Training complete.")
    logger.info("Results directory: %s", results.save_dir)
    logger.info("Best weights: %s", Path(results.save_dir) / "weights" / "best.pt")


if __name__ == "__main__":
    main()
