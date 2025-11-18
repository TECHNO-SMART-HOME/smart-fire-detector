"""Live webcam inference utility for the smart fire detector YOLO model."""
from __future__ import annotations
import argparse
import logging
import multiprocessing
import threading
from pathlib import Path
from typing import Iterable

import torch
from ultralytics import YOLO

# --- IMPORT YOUR NEW MODULE ---
import fire_alert_system 

try:
    import cv2
except ImportError as exc:
    raise SystemExit("OpenCV required. 'pip install opencv-python'") from exc

try:
    from playsound import playsound
except ImportError:
    playsound = None

PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "runs"
WINDOW_NAME = "Smart Fire Detector"
DEFAULT_ALARM_SOUND = PROJECT_ROOT / "firealarm.mp3"

def discover_default_weights() -> Path:
    def iter_candidates() -> Iterable[Path]:
        if RUNS_DIR.exists():
            for path in sorted(RUNS_DIR.rglob("weights/best.pt"), key=lambda c: c.stat().st_mtime, reverse=True):
                yield path
        for filename in ("yolo11n.pt", "yolov8s.pt"):
            path = PROJECT_ROOT / filename
            if path.exists(): yield path
    for candidate in iter_candidates(): return candidate
    raise FileNotFoundError("No checkpoints (best.pt / yolo11n.pt) found.")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smart Fire Detector")
    parser.add_argument("--weights", type=Path, default=discover_default_weights())
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--fire-alert-threshold", type=float, default=0.85)
    parser.add_argument("--alarm-sound", type=Path, default=DEFAULT_ALARM_SOUND)
    # Standard args
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hide-labels", action="store_true")
    parser.add_argument("--hide-conf", action="store_true")
    return parser.parse_args()

class AlarmPlayer:
    def __init__(self, sound_path: Path | None, logger: logging.Logger) -> None:
        self.sound_path = sound_path
        self.logger = logger
        self._lock = threading.Lock()
        self._is_playing = False
        if self.sound_path and not self.sound_path.exists():
            self.logger.warning(f"Sound file not found: {self.sound_path}")
            self.sound_path = None

    def trigger(self) -> None:
        if not self.sound_path or not playsound: return
        with self._lock:
            if self._is_playing: return
            self._is_playing = True
        threading.Thread(target=self._play_sound, daemon=True).start()

    def _play_sound(self) -> None:
        try: playsound(str(self.sound_path))
        except Exception as e: self.logger.error(f"Audio error: {e}")
        finally: 
            with self._lock: self._is_playing = False

def run_inference_loop(model, source, conf, fire_thresh, logger, alarm, mirror, imgsz, hide_lbl, hide_conf):
    capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not capture.isOpened(): raise RuntimeError(f"Cannot open source '{source}'")
    
    logger.info("Starting Preview. Press 'q' to exit.")
    frame_idx = 0

    try:
        while True:
            success, frame = capture.read()
            if not success: break
            if mirror: frame = cv2.flip(frame, 1)

            results = model.predict(source=frame, imgsz=imgsz, conf=conf, verbose=False)
            result = results[0]
            
            fire_detected = False
            max_conf = 0.0

            # Logic to find 'fire' class
            if result.boxes and result.boxes.cls is not None:
                # Filter boxes
                names = result.names
                cls_ids = result.boxes.cls.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                
                for i, cls_id in enumerate(cls_ids):
                    label = names[int(cls_id)].lower()
                    score = confs[i]
                    
                    if label == "fire":
                        fire_detected = True
                        if score > max_conf: max_conf = score

            # Draw boxes
            annotated = result.plot(labels=not hide_lbl, conf=not hide_conf)

            # --- ACTION BLOCK ---
            if fire_detected and max_conf >= fire_thresh:
                # 1. Play Alarm
                if alarm: alarm.trigger()
                
                # 2. Blink Screen Text
                if (frame_idx // 10) % 2 == 0:
                    cv2.putText(annotated, "FIRE DETECTED", (20, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
                # 3. CALL YOUR NEW MODULE (One line only)
                fire_alert_system.send_alert(max_conf)

            cv2.imshow(WINDOW_NAME, annotated)
            frame_idx += 1
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        capture.release()
        cv2.destroyWindow(WINDOW_NAME)

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger("FireDetector")
    args = parse_args()

    logger.info(f"Loading Model: {args.weights}")
    model = YOLO(str(args.weights))
    
    alarm = AlarmPlayer(args.alarm_sound, logger)

    run_inference_loop(
        model, args.source, args.conf, args.fire_alert_threshold, 
        logger, alarm, args.mirror, args.imgsz, args.hide_labels, args.hide_conf
    )

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()