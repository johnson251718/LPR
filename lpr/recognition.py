"""Text recognition utilities for license plate crops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import cv2
import easyocr
import numpy as np


@dataclass
class PlateReading:
    """Container holding recognition details for a single plate."""

    bbox: Tuple[int, int, int, int]
    text: str
    confidence: float


class LicensePlateRecognizer:
    """Recognizes alphanumeric content from detected license plate regions."""

    def __init__(
        self,
        languages: Sequence[str] | None = None,
        gpu: bool = False,
    ) -> None:
        self.languages = list(languages or ["en"])
        self._reader = easyocr.Reader(self.languages, gpu=gpu)

    def recognize(
        self, frame: np.ndarray, boxes: Iterable[Tuple[int, int, int, int]]
    ) -> List[PlateReading]:
        """Return best-effort text predictions for the provided bounding boxes."""

        readings: List[PlateReading] = []
        for (x, y, w, h) in boxes:
            crop = frame[y : y + h, x : x + w]
            if crop.size == 0:
                continue

            processed = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            processed = cv2.bilateralFilter(processed, 11, 17, 17)
            processed = cv2.equalizeHist(processed)

            results = self._reader.readtext(processed)
            if not results:
                continue

            best = max(results, key=lambda item: float(item[2]))
            text = best[1].upper().replace(" ", "")
            confidence = float(best[2])
            readings.append(PlateReading((x, y, w, h), text, confidence))

        return readings


__all__ = ["LicensePlateRecognizer", "PlateReading"]
