"""Utilities for locating potential license plates in images."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np


class LicensePlateDetector:
    """Detects potential license plate regions using a Haar cascade."""

    def __init__(
        self,
        cascade_path: Path | str | None = None,
        *,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (30, 30),
    ) -> None:
        self.cascade_path = (
            Path(cascade_path)
            if cascade_path is not None
            else Path(__file__).resolve().parent.parent
            / "resources"
            / "haarcascade_russian_plate_number.xml"
        )
        if not self.cascade_path.exists():
            raise FileNotFoundError(
                "Cascade file not found. Download it from the OpenCV repository "
                "and place it at resources/haarcascade_russian_plate_number.xml"
            )

        self._classifier = cv2.CascadeClassifier(str(self.cascade_path))
        if self._classifier.empty():
            raise ValueError(
                "Failed to load cascade classifier. Ensure the XML file is valid."
            )

        self.scale_factor = float(scale_factor)
        self.min_neighbors = int(min_neighbors)
        self.min_size = tuple(int(value) for value in min_size)

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Return bounding boxes for detected license plates.

        Args:
            frame: Image array in BGR color order.

        Returns:
            List of bounding boxes in (x, y, width, height) format.
        """

        if frame is None or frame.size == 0:
            raise ValueError("frame must be a non-empty numpy array")

        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayscale = cv2.equalizeHist(grayscale)

        detections: Sequence[Tuple[int, int, int, int]] = self._classifier.detectMultiScale(
            grayscale,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )
        return [tuple(map(int, detection)) for detection in detections]


__all__ = ["LicensePlateDetector"]
