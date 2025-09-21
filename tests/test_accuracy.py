"""Accuracy regression tests for the license plate recognition pipeline."""

from __future__ import annotations

import random
import string
from pathlib import Path

import pytest

cv2 = pytest.importorskip(
    "cv2", reason="opencv-python is required for the accuracy benchmark"
)
np = pytest.importorskip("numpy", reason="numpy is required for the accuracy benchmark")

from lpr.pipeline import PlateRecognitionPipeline


random.seed(1234)


def _generate_plate_image(text: str, width: int = 360, height: int = 180) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    background_color = tuple(int(value) for value in rng.integers(30, 90, size=3))
    image = np.full((height, width, 3), background_color, dtype=np.uint8)

    margin_x = int(0.08 * width)
    margin_y = int(0.18 * height)
    top_left = (margin_x, margin_y)
    bottom_right = (width - margin_x, height - margin_y)
    cv2.rectangle(image, top_left, bottom_right, (235, 235, 235), thickness=-1)
    cv2.rectangle(image, top_left, bottom_right, (20, 20, 20), thickness=3)

    font_scale = 1.3
    thickness = 3
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    plate_width = bottom_right[0] - top_left[0]
    plate_height = bottom_right[1] - top_left[1]
    text_x = top_left[0] + (plate_width - text_size[0]) // 2
    text_y = top_left[1] + (plate_height + text_size[1]) // 2 - 4
    cv2.putText(
        image,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (10, 10, 10),
        thickness,
        cv2.LINE_AA,
    )

    noise = rng.normal(0, 8, size=image.shape).astype(np.int16)
    noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy


def _create_plate_text() -> str:
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # avoid ambiguous characters
    digits = string.digits
    prefix = "".join(random.choice(letters) for _ in range(3))
    suffix = "".join(random.choice(digits) for _ in range(3))
    return f"{prefix}{suffix}"


def test_synthetic_plate_accuracy(tmp_path: Path) -> None:
    pipeline = PlateRecognitionPipeline()

    samples = 20
    successes = 0
    for index in range(samples):
        text = _create_plate_text()
        image = _generate_plate_image(text)
        image_path = tmp_path / f"plate_{index}.png"
        cv2.imwrite(str(image_path), image)

        readings = pipeline.process_image(image_path)
        if not readings:
            continue
        best = max(readings, key=lambda reading: reading.confidence)
        if best.text == text:
            successes += 1

    accuracy = successes / samples
    assert accuracy >= 0.9, f"expected >= 0.9 accuracy, got {accuracy:.2f}"
