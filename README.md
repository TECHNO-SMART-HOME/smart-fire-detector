# Smart Fire Detector

## Overview
This repository hosts a YOLO-powered computer vision prototype that spots fire and smoke in real time. It bundles:
- `webcam_inference.py` – a live preview loop that highlights detections, mirrors consumer smart-home camera behavior, and plays an audible alarm when a confident fire signal appears.
- `trainer.py` – a GPU-first training harness that resolves dataset paths, resumes Ultralytics runs, and materializes a `data.resolved.yaml` for deterministic experiments.
- Roboflow-exported dataset stubs (`train/`, `valid/`, `test/` and `data.yaml`) plus example checkpoints (`yolo11n.pt`, `yolov8s.pt`) so you can fine-tune or evaluate immediately.

## Purpose
The codebase is meant as one of the early prototypes for a smart-home fire sensor:
- Validate whether RGB cameras can complement traditional flame/heat sensors before investing in custom hardware.
- Provide a reference implementation teams can wrap into a microservice, Home Assistant add-on, or edge gateway.
- Accelerate experimentation on alarm logic (e.g., what confidence threshold triggers sirens, when to mirror frames, how to log alerts).

## Usage
### 1. Environment setup
1. Ensure Python 3.10+ and a CUDA-capable GPU (trainer requires CUDA; inference can fall back to CPU).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Drop any additional YOLO `.pt` checkpoints beside this README or under `runs/*/weights/`.

### 2. Run the webcam prototype
```bash
python webcam_inference.py \
  --source 0 \
  --mirror \
  --fire-alert-threshold 0.85 \
  --alarm-sound firealarm.mp3
```
- `--source` accepts a camera index, video path, or RTSP/HTTP stream.
- The script auto-selects the freshest `runs/*/weights/best.pt`; override with `--weights path/to/model.pt`.
- Audio alerts use `playsound`; set `--alarm-sound` to a custom MP3/WAV or `--alarm-sound ""` to disable playback.
- Press `q` or `Esc` to exit the preview window.

### 3. Train or fine-tune a model
```bash
python trainer.py \
  --data data.yaml \
  --weights yolov8s.pt \
  --epochs 50 \
  --batch 16 \
  --project runs \
  --name smart-flame-detector
```
- The trainer resolves the relative paths inside `data.yaml` and writes `data.resolved.yaml` automatically for Ultralytics.
- `--device` defaults to GPU `0`; pass `--device 0,1` for multi-GPU training.
- Checkpoints land under `runs/<name>/weights/`; pick `best.pt` for deployment and copy it beside `webcam_inference.py` if desired.

### 4. Integrate with a smart-home stack
While this repository focuses on detection, the outputs are designed for downstream automation:
- Listen to the log stream (stdout) or extend `AlarmPlayer` to call MQTT/webhooks when `fire_alert_threshold` is met.
- Wrap `webcam_inference.py` in a service that exposes the annotated frames over RTSP/WebRTC for monitoring panels.
- Export Ultralytics results to ONNX/TensorRT if the final hardware target is an embedded edge device.

## Dataset note
`data.yaml` points to the Roboflow dataset `firesensor-zwryw/fire-h2gkf-6mcdq` (CC BY 4.0). Replace the `train/`, `valid/`, and `test/` directories with your own captures to adapt the prototype to new environments or camera placements.

---
Use this README as the living blueprint for transforming the prototype into a production-ready smart-home fire sensor. Update it as you iterate on alarm logic, deployment targets, or dataset sources.
