"""Utilities for locating potential license plates in images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import cv2
import numpy as np


@dataclass
class DetectorConfig:
    """Configuration values for :class:`LicensePlateDetector`."""

    min_plate_area: float = 4500.0
    aspect_ratio_range: Tuple[float, float] = (2.0, 6.0)
    morph_kernel_size: Tuple[int, int] = (5, 5)
    max_candidates: int = 5


class LicensePlateDetector:
    """Detects potential license plate regions using contour analysis."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if frame is None or frame.size == 0:
            raise ValueError("frame must be a non-empty numpy array")

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = cv2.subtract(grad_x, grad_y)
        gradient = cv2.convertScaleAbs(gradient)
        gradient = cv2.morphologyEx(
            gradient,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, self.config.morph_kernel_size),
        )
        _, binary = cv2.threshold(
            gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, self.config.morph_kernel_size),
            iterations=2,
        )
        binary = cv2.erode(binary, None, iterations=1)
        binary = cv2.dilate(binary, None, iterations=1)
        return binary

    def _filter_candidates(
        self, contours: Iterable[np.ndarray], frame_shape: Tuple[int, int, int]
    ) -> List[Tuple[int, int, int, int]]:
        candidates: List[Tuple[int, int, int, int]] = []
        height, width = frame_shape[:2]
        min_area = self.config.min_plate_area
        min_w, min_h = 0.12 * width, 0.08 * height
        max_w, max_h = 0.95 * width, 0.6 * height
        min_ratio, max_ratio = self.config.aspect_ratio_range

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w < min_w or h < min_h or w > max_w or h > max_h:
                continue
            ratio = w / float(h)
            if ratio < min_ratio or ratio > max_ratio:
                continue
            candidates.append((x, y, w, h))

        candidates.sort(key=lambda box: box[2] * box[3], reverse=True)
        if self.config.max_candidates:
            candidates = candidates[: self.config.max_candidates]
        return candidates

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Return bounding boxes for detected license plates."""

        self._validate_frame(frame)

        processed = self._preprocess(frame)
        contours, _ = cv2.findContours(
            processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        boxes = self._filter_candidates(contours, frame.shape)
        if boxes:
            return boxes

        height, width = frame.shape[:2]
        return [(0, 0, width, height)]


__all__ = ["LicensePlateDetector"]
