# License Plate Recognition (LPR)

This project implements a modular license plate recognition (LPR) pipeline in Python.
Out of the box it ships with a contour-based detector and template recognizer that run
fully offline, and it can be upgraded to near-YOLO accuracy by plugging in ONNX exports
of modern detectors and recognizers—without depending on the Ultralytics runtime.

## Features

- **Dual detection backends** – keep the lightweight contour detector for quick tests
  or switch to a YOLO ONNX model powered by `cv2.dnn` for state-of-the-art plate recall.
- **Flexible recognition** – default template matching provides zero-dependency OCR
  while an optional CRNN/LPRNet recognizer (via `onnxruntime`) boosts transcription
  accuracy on challenging plates.
- **Image and video CLI** – process photographs or dashcam footage with optional
  annotation outputs.
- **Regression-tested quality** – a synthetic benchmark enforces ≥ 90 % accuracy to
  guard against accidental regressions.

## Requirements

- Python 3.10 or newer
- [OpenCV](https://opencv.org/) and [NumPy](https://numpy.org/)
- [PyTest](https://docs.pytest.org/) for the regression suite
- [ONNX Runtime](https://onnxruntime.ai/) *(optional — required only when using the
  CRNN recognizer)*

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Downloading pretrained models (optional)

The classical contour/template stack works without extra assets. To reach YOLO-level
performance you should provide pretrained ONNX models:

1. **YOLO detector** – download or export a license-plate detector to ONNX (e.g. a
   YOLOv5/YOLOv8 model trained on plates). Place the file under `resources/`:

   ```bash
   curl -L -o resources/yoloplate.onnx https://example.com/path/to/license-plate.onnx
   ```

   Any ONNX model that outputs `[x, y, w, h, obj, class_scores…]` tensors is supported.

2. **CRNN/LPRNet recognizer** – obtain an ONNX transcription model trained on license
   plates (for example an exported LPRNet or CRNN checkpoint):

   ```bash
   curl -L -o resources/lprnet.onnx https://example.com/path/to/lprnet.onnx
   ```

   Ensure you know the alphabet the model was trained on so you can pass it to the CLI.

## Usage

The command-line interface is exposed via `python -m lpr.cli`. Run `--help` to view all
options.

### Contour + template pipeline (default)

```bash
python -m lpr.cli \
  --image path/to/car.jpg \
  --output annotated.jpg
```

### YOLO detector + CRNN recognizer

```bash
python -m lpr.cli \
  --image path/to/car.jpg \
  --detector yolo \
  --yolo-model resources/yoloplate.onnx \
  --recognizer crnn \
  --crnn-model resources/lprnet.onnx \
  --crnn-alphabet 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ \
  --output annotated.jpg
```

The same arguments work with `--video` and optional `--max-frames` when processing
dashcam footage.

### Notable CLI parameters

- `--detector` – choose between `contour` (default) and `yolo` ONNX detection.
- `--max-candidates` – number of plate candidates per frame (default: 5).
- `--yolo-*` – input size, confidence thresholds, and allowed class IDs for YOLO.
- `--recognizer` – select `template` (default) or `crnn`.
- `--min-character-score` – minimum template correlation per character.
- `--crnn-*` – alphabet, input geometry, and thresholds for the CRNN recognizer.

## Accuracy evaluation

Run the regression suite to confirm the default pipeline maintains ≥ 90 % accuracy on
synthetic plates:

```bash
python -m pytest
```

The test harness deterministically generates 20 noisy plate images and asserts that at
least 18 are decoded correctly.

## Project structure

```
.
├── lpr
│   ├── __init__.py            # Public package exports
│   ├── cli.py                 # Command-line interface & configuration wiring
│   ├── detection.py           # Contour and YOLO detectors
│   ├── pipeline.py            # High-level orchestration helpers
│   └── recognition.py         # Template and CRNN recognizers
├── requirements.txt           # Runtime (and test) dependencies
├── resources/                 # Place downloaded ONNX models here (optional)
├── tests
│   └── test_accuracy.py       # ≥90% accuracy regression test
└── README.md                  # Project documentation
```

## Next steps

- Add multi-frame tracking to stabilize detections in video streams.
- Integrate data-driven post-processing (e.g. region-specific plate formatting).
- Expose the pipeline via a REST API or a lightweight UI for rapid validation.
