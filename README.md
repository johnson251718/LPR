# License Plate Recognition (LPR)

This project implements a Python-based license plate recognition (LPR) pipeline that
relies purely on classical computer vision. A contour-driven detector extracts plate
regions, while a lightweight template matcher decodes the alphanumeric content without
requiring any Ultralytics components or heavyweight OCR dependencies.

## Features

- **Robust detection** – blackhat-enhanced gradients with adaptive kernels isolate
  rectangular plate candidates without needing external XML cascades.
- **Template-based recognition** – characters are segmented and matched against
  auto-generated templates, enabling fully offline operation.
- **Image and video CLI** – process photographs or dashcam footage, optionally writing
  annotated outputs to disk.
- **Reproducible accuracy checks** – synthetic regression data validates that the
  recognizer exceeds 90 % accuracy before distribution.

## Requirements

- Python 3.10 or newer
- [OpenCV](https://opencv.org/) and [NumPy](https://numpy.org/)
- [PyTest](https://docs.pytest.org/) for running the automated accuracy check

Install the Python dependencies and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No additional model downloads are required—the detector and recognizer run entirely on
the packaged code.

## Usage

The command-line interface is exposed via `python -m lpr.cli`. Use `--help` to inspect
all options.

### Image recognition

```bash
python -m lpr.cli \
  --image path/to/car.jpg \
  --output annotated.jpg
```

Detected plates are printed to stdout, and an annotated copy is written when `--output`
is supplied.

### Video recognition

```bash
python -m lpr.cli \
  --video path/to/dashcam.mp4 \
  --output annotated.mp4 \
  --max-frames 300
```

Each frame is processed independently, and the optional `--max-frames` cap lets you
limit work for quick experiments.

### Tunable thresholds

- `--max-candidates` controls how many plate hypotheses per frame are sent to the
  recognizer (default: 5).
- `--min-character-score` sets the minimum normalized correlation score required to
  accept individual character matches (default: 0.5).
- Programmatic users can tweak `DetectorConfig` (area ratios, kernel sizes, padding)
  when instantiating `LicensePlateDetector` for especially small or skewed plates.

## Accuracy evaluation

Run the regression suite to verify that the pipeline maintains ≥ 90 % accuracy on the
synthetic benchmark set:

```bash
python -m pytest
```

The test harness procedurally generates 20 plate images with varied backgrounds and
noise, ensuring deterministic coverage of typical alphanumeric layouts. The assertion
fails if fewer than 18 plates are decoded correctly.

## Project structure

```
.
├── lpr
│   ├── __init__.py            # Public package exports
│   ├── cli.py                 # Command-line interface
│   ├── detection.py           # Contour-driven plate detector
│   ├── pipeline.py            # High-level orchestration helpers
│   └── recognition.py         # Template-based OCR implementation
├── requirements.txt           # Runtime (and test) dependencies
├── tests
│   └── test_accuracy.py       # ≥90% accuracy regression test
└── README.md                  # Project documentation
```

## Next steps

Potential extensions include:

- Incorporating plate tracking across consecutive video frames to stabilize results.
- Learning a small convolutional classifier for broader font coverage while preserving
  offline execution.
- Exposing the pipeline through a REST API or lightweight web interface.
