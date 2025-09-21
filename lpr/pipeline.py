"""Pipeline orchestration for license plate recognition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import cv2
import numpy as np

from .detection import LicensePlateDetector
from .recognition import LicensePlateRecognizer, PlateReading


@dataclass
class FrameReadings:
    """Recognition results for a single frame in a video."""

    frame_index: int
    readings: Sequence[PlateReading]


class PlateRecognitionPipeline:
    """High level helper that wires together detection and recognition."""

    def __init__(
        self,
        detector: LicensePlateDetector | None = None,
        recognizer: LicensePlateRecognizer | None = None,
    ) -> None:
        self.detector = detector or LicensePlateDetector()
        self.recognizer = recognizer or LicensePlateRecognizer()

    @staticmethod
    def _annotate(frame: np.ndarray, readings: Iterable[PlateReading]) -> np.ndarray:
        annotated = frame.copy()
        for reading in readings:
            x, y, w, h = reading.bbox
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{reading.text} ({reading.confidence:.2f})"
            cv2.putText(
                annotated,
                label,
                (x, y - 10 if y - 10 > 10 else y + h + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
        return annotated

    def process_image(
        self,
        image_path: str | Path,
        *,
        output_path: str | Path | None = None,
    ) -> List[PlateReading]:
        """Run detection and recognition on an image file."""

        image_path = Path(image_path)
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(f"Unable to read image at {image_path!s}")

        boxes = self.detector.detect(frame)
        readings = self.recognizer.recognize(frame, boxes)

        if output_path:
            annotated = self._annotate(frame, readings)
            cv2.imwrite(str(output_path), annotated)

        return readings

    def process_video(
        self,
        video_path: str | Path,
        *,
        output_path: str | Path | None = None,
        max_frames: int | None = None,
    ) -> List[FrameReadings]:
        """Run detection and recognition on a video file."""

        video_path = Path(video_path)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Unable to open video at {video_path!s}")

        writer = None
        if output_path is not None:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        results: List[FrameReadings] = []
        frame_index = 0
        while True:
            success, frame = capture.read()
            if not success:
                break

            boxes = self.detector.detect(frame)
            readings = self.recognizer.recognize(frame, boxes)
            results.append(FrameReadings(frame_index, tuple(readings)))

            if writer is not None:
                annotated = self._annotate(frame, readings)
                writer.write(annotated)

            frame_index += 1
            if max_frames is not None and frame_index >= max_frames:
                break

        capture.release()
        if writer is not None:
            writer.release()

        return results


__all__ = ["PlateRecognitionPipeline", "FrameReadings"]
