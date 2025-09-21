# License Plate Recognition (LPR)

This project provides a minimal, end-to-end license plate recognition (LPR) pipeline
implemented in Python. It demonstrates how to combine classical computer vision for
plate detection with OCR-based text recognition to extract the characters displayed on
a vehicle license plate. The code base intentionally avoids any Ultralytics
dependencies, relying instead on OpenCV and EasyOCR.

## Features

- Detect license plates in still images or videos using an OpenCV Haar cascade.
- Recognize alphanumeric content from detected plates with EasyOCR.
- Command-line interface that can annotate outputs and prints textual results.
- Modular architecture that separates detection, recognition, and pipeline orchestration
  for easy extension.

## Requirements

- Python 3.10 or newer.
- [OpenCV](https://opencv.org/) and [EasyOCR](https://github.com/JaidedAI/EasyOCR)
  Python packages.
- A Haar cascade XML file for license plate detection. The pipeline expects the file at
  `resources/haarcascade_russian_plate_number.xml` (see below).

Install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the OpenCV Haar cascade for license plates:

```bash
curl -L -o resources/haarcascade_russian_plate_number.xml \
  https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_russian_plate_number.xml
```

> **Note:** If direct downloads are blocked, manually retrieve the XML file from the
> OpenCV repository and place it in the `resources/` directory.

## Usage

The project exposes a CLI entry point through `python -m lpr.cli`. Run
`python -m lpr.cli --help` to inspect all available options.

### Image recognition

```bash
python -m lpr.cli \
  --image path/to/car.jpg \
  --output annotated.jpg
```

The command prints any detected plate texts to the console and, if `--output` is
provided, writes an annotated copy of the image showing the detections.

### Video recognition

```bash
python -m lpr.cli \
  --video path/to/dashcam.mp4 \
  --output annotated.mp4 \
  --max-frames 300
```

The pipeline processes each frame, aggregates detections, and optionally writes an
annotated video. Use `--max-frames` during testing to restrict the processing window.

### Advanced options

- `--languages` lets you specify a comma-separated list of EasyOCR language codes
  (default: `en`).
- `--gpu` toggles GPU acceleration in EasyOCR when a supported GPU is available.
- `--cascade-path` allows pointing to a custom Haar cascade XML file.

## Project structure

```
.
├── lpr
│   ├── cli.py                # Command-line interface
│   ├── detection.py          # Haar cascade detection utilities
│   ├── pipeline.py           # High-level pipeline orchestration
│   └── recognition.py        # EasyOCR text recognition helpers
├── resources                 # Expected location for cascade files and assets
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## Next steps

This repository is intentionally lightweight and can serve as a starting point for
more advanced LPR systems. Potential enhancements include:

- Training a custom object detector tailored to a specific region or plate format.
- Incorporating plate tracking across video frames to increase stability.
- Adding a REST API or web interface to expose the recognition pipeline as a service.
