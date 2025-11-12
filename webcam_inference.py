"""Live webcam inference utility for the smart flame detector YOLO model."""

from __future__ import annotations

import argparse
import logging
import multiprocessing
from pathlib import Path
from typing import Iterable

import torch
from ultralytics import YOLO

try:
    import cv2
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "OpenCV is required for webcam streaming. Install it with 'pip install opencv-python'."
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "runs"
WINDOW_NAME = "Smart Flame Detector"


def discover_default_weights() -> Path:
    """Return the newest best.pt from runs/, falling back to packaged checkpoints."""

    def iter_candidates() -> Iterable[Path]:
        if RUNS_DIR.exists():
            for path in sorted(
                RUNS_DIR.rglob("weights/best.pt"),
                key=lambda candidate: candidate.stat().st_mtime,
                reverse=True,
            ):
                yield path

        # Fallback to checkpoints tracked in the repo root.
        for filename in ("yolo11n.pt", "yolov8s.pt"):
            path = PROJECT_ROOT / filename
            if path.exists():
                yield path

    for candidate in iter_candidates():
        return candidate

    raise FileNotFoundError(
        "Could not locate any checkpoint. Expected a runs/*/weights/best.pt or "
        "one of yolo11n.pt / yolov8s.pt to exist beside this script."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the trained smart flame detector model on a webcam feed."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=discover_default_weights(),
        help="Path to the YOLO checkpoint (.pt). Defaults to the most recent runs/*/weights/best.pt.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device identifier. Defaults to GPU 0 when available, otherwise CPU.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Camera index or video source path/URL. Use '0' for the default webcam.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Confidence threshold for detections."
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Horizontally flip frames before inference (useful for front-facing webcams).",
    )
    parser.add_argument(
        "--hide-labels",
        action="store_true",
        help="Hide class labels in the rendered preview.",
    )
    parser.add_argument(
        "--hide-conf",
        action="store_true",
        help="Hide confidence values in the rendered preview.",
    )
    return parser.parse_args()


def coerce_source(source: str) -> int | str:
    """Interpret numeric camera indices while preserving literal paths/URLs."""
    try:
        return int(source)
    except ValueError:
        return source


def select_device(requested: str | None) -> str:
    """Pick a device for inference, preferring CUDA when available."""
    if requested:
        return requested

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return "0"

    return "cpu"


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("webcam_inference")


def run_inference_loop(
    model: YOLO,
    source: int | str,
    device: str,
    imgsz: int,
    conf: float,
    mirror: bool,
    hide_labels: bool,
    hide_conf: bool,
    logger: logging.Logger,
) -> None:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source '{source}'.")

    logger.info("Press 'q' or ESC to exit the preview window.")

    try:
        while True:
            success, frame = capture.read()
            if not success:
                logger.warning("Failed to read from source — exiting preview loop.")
                break

            if mirror:
                frame = cv2.flip(frame, 1)

            results = model.predict(
                source=frame,
                device=device,
                imgsz=imgsz,
                conf=conf,
                verbose=False,
            )
            annotated = results[0].plot(labels=not hide_labels, conf=not hide_conf)
            cv2.imshow(WINDOW_NAME, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        capture.release()
        cv2.destroyWindow(WINDOW_NAME)


def main() -> None:
    logger = setup_logging()
    args = parse_args()
    device = select_device(args.device)

    if not args.weights.exists():
        raise FileNotFoundError(f"Checkpoint not found at '{args.weights}'.")

    logger.info("Loading model weights from %s", args.weights)
    model = YOLO(str(args.weights))

    run_inference_loop(
        model=model,
        source=coerce_source(args.source),
        device=device,
        imgsz=args.imgsz,
        conf=args.conf,
        mirror=args.mirror,
        hide_labels=args.hide_labels,
        hide_conf=args.hide_conf,
        logger=logger,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
