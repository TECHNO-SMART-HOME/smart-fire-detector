"""Trainer script for the smart flame detector dataset using Ultralytics YOLO."""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
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
        "--cache",
        action="store_true",
        help="Cache images for faster training (requires enough CPU RAM).",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume the most recent run for the selected project/name pair."
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


def choose_device(preferred: str | None) -> str:
    """Require CUDA-backed execution and provide actionable diagnostics when unavailable."""

    def cuda_ready() -> bool:
        if not torch.cuda.is_available():
            return False
        return torch.cuda.device_count() > 0

    def cuda_diagnostic() -> str:
        build = torch.version.cuda
        if build is None:
            return (
                "PyTorch was installed without CUDA support (torch.version.cuda is None). "
                "Install a CUDA-enabled build from https://pytorch.org/get-started/locally/ to use the GPU."
            )
        if not torch.cuda.is_available():
            return (
                f"PyTorch reports CUDA build {build}, but torch.cuda.is_available() returned False. "
                "Verify that NVIDIA drivers and the matching CUDA runtime are installed."
            )
        if torch.cuda.device_count() == 0:
            return (
                f"PyTorch reports CUDA build {build}, but torch.cuda.device_count() returned 0 GPUs. "
                "Ensure at least one CUDA-capable GPU is accessible to the OS."
            )
        return (
            f"PyTorch reports CUDA build {build}, but the requested CUDA device could not be initialized. "
            "Verify that the GPU identifier exists and is not in exclusive use."
        )

    if preferred:
        normalized = preferred.strip().lower()
        if normalized in {"cpu", "cpu:0"}:
            raise RuntimeError("CPU execution is disabled. Select a CUDA device such as --device 0.")
        if not cuda_ready():
            diagnostic = cuda_diagnostic()
            raise RuntimeError(f"{diagnostic} Requested device '{preferred}' but CUDA is unavailable.")
        return preferred

    if cuda_ready():
        return "0"

    diagnostic = cuda_diagnostic()
    raise RuntimeError(f"{diagnostic} GPU execution is mandatory for this trainer.")


def main() -> None:
    """Entrypoint for training. freeze_support guard avoids Windows spawn issues."""
    logger = configure_logging()
    args = parse_args()
    logger.info("Starting training run with args: %s", args)
    data_yaml = Path(args.data)
    if not os.path.exists(os.fspath(data_yaml)):
        logger.error("Dataset config not found at '%s'. Ensure the YAML file exists.", data_yaml)
        raise SystemExit(1)
    logger.info("Validated dataset config at %s", data_yaml)
    data_cfg = materialize_data_config(args.data, logger)
    device = choose_device(args.device)
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
    multiprocessing.freeze_support()
    main()
