from types import SimpleNamespace

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from lpr.detection import YoloLicensePlateDetector


class FakeSession:
    def __init__(self):
        self._inputs = [SimpleNamespace(name="images")]
        self._outputs = [SimpleNamespace(name="output0")]

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, output_names, feeds):  # pragma: no cover - not used in tests
        raise RuntimeError("FakeSession does not execute models")


def test_decode_predictions_supports_yolov5_and_yolov8_layouts():
    detector = YoloLicensePlateDetector(
        "fake.onnx", conf_threshold=0.5, iou_threshold=0.4, session=FakeSession()
    )

    yolov5_like = [
        [50.0, 50.0, 20.0, 10.0, 0.9, 0.8],
        [30.0, 30.0, 15.0, 12.0, 0.95, 0.2],
        [45.0, 45.0, 18.0, 10.0, 0.4, 0.9],
        [90.0, 90.0, 40.0, 36.0, 0.3, 0.8],
    ]
    detections_v5 = detector._decode_predictions(
        yolov5_like, original_shape=(100, 100), scale=1.0, pad=(0.0, 0.0)
    )

    assert len(detections_v5) == 1
    det_v5 = detections_v5[0]
    assert det_v5.box == (40, 45, 60, 55)
    assert det_v5.class_id == 0
    assert det_v5.score == pytest.approx(0.72, abs=1e-6)

    yolov8_like = [
        [
            [80.0, 84.0],
            [80.0, 84.0],
            [30.0, 32.0],
            [30.0, 32.0],
            [0.9, 0.6],
            [0.7, 0.9],
        ]
    ]

    detections_v8 = detector._decode_predictions(
        yolov8_like, original_shape=(120, 120), scale=1.0, pad=(0.0, 0.0)
    )

    assert len(detections_v8) == 1
    det_v8 = detections_v8[0]
    assert det_v8.class_id == 0
    assert det_v8.score == pytest.approx(0.63, abs=1e-6)
    assert det_v8.box[0] <= det_v8.box[2] and det_v8.box[1] <= det_v8.box[3]
