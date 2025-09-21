"""Utilities for locating potential license plates in images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import cv2
import numpy as np


@dataclass
class DetectorConfig:
    """Configuration values for :class:`LicensePlateDetector`."""

    min_plate_area: float = 1200.0
    min_plate_area_ratio: float = 0.0005
    max_plate_area_ratio: float = 0.8
    aspect_ratio_range: Tuple[float, float] = (2.0, 6.5)
    min_solidity: float = 0.4
    padding_ratio: float = 0.03
    blackhat_kernel_size: Tuple[int, int] = (21, 7)
    closing_kernel_size: Tuple[int, int] = (25, 7)
    square_kernel_size: Tuple[int, int] = (5, 5)
    blur_kernel_size: Tuple[int, int] = (5, 5)
    max_candidates: int = 5


class LicensePlateDetector:
    """Detects potential license plate regions using contour analysis."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if frame is None or frame.size == 0:
            raise ValueError("frame must be a non-empty numpy array")

    @staticmethod
    def _scale_dimension(value: int, scale: float, *, minimum: int = 3) -> int:
        scaled = max(minimum, int(round(value * scale)))
        if scaled % 2 == 0:
            scaled += 1
        return scaled

    def _make_kernel(
        self, base_size: Tuple[int, int], scale: float, *, minimum: int = 3
    ) -> np.ndarray:
        width = self._scale_dimension(base_size[0], scale, minimum=minimum)
        height = self._scale_dimension(base_size[1], scale, minimum=minimum)
        return cv2.getStructuringElement(cv2.MORPH_RECT, (width, height))

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        height, width = gray.shape[:2]
        scale = max(0.5, min(height, width) / 400.0)

        rect_kernel = self._make_kernel(self.config.blackhat_kernel_size, scale)
        close_kernel = self._make_kernel(self.config.closing_kernel_size, scale)
        square_kernel = self._make_kernel(self.config.square_kernel_size, scale)

        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)
        grad_x = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=3)
        grad_x = np.absolute(grad_x)
        min_val, max_val = float(np.min(grad_x)), float(np.max(grad_x))
        if max_val > min_val:
            grad_x = (255.0 * (grad_x - min_val) / (max_val - min_val)).astype(np.uint8)
        else:
            grad_x = np.zeros_like(grad_x, dtype=np.uint8)

        grad_x = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, close_kernel)
        blur_width = self._scale_dimension(self.config.blur_kernel_size[0], scale)
        blur_height = self._scale_dimension(self.config.blur_kernel_size[1], scale)
        grad_x = cv2.GaussianBlur(grad_x, (blur_width, blur_height), 0)
        _, binary = cv2.threshold(
            grad_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )

        light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, rect_kernel)
        light = cv2.normalize(light, None, 0, 255, cv2.NORM_MINMAX)
        _, light_thresh = cv2.threshold(
            light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        binary = cv2.bitwise_and(binary, light_thresh)

        binary = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, square_kernel, iterations=2
        )
        binary = cv2.erode(binary, None, iterations=1)
        binary = cv2.dilate(binary, None, iterations=2)
        return binary

    def _filter_candidates(
        self, contours: Iterable[np.ndarray], frame_shape: Tuple[int, int, int]
    ) -> List[Tuple[int, int, int, int]]:
        candidates: List[Tuple[int, int, int, int]] = []
        height, width = frame_shape[:2]
        frame_area = float(width * height)
        min_area = max(
            self.config.min_plate_area, self.config.min_plate_area_ratio * frame_area
        )
        max_area = (
            self.config.max_plate_area_ratio * frame_area
            if 0 < self.config.max_plate_area_ratio <= 1.0
            else frame_area
        )
        min_ratio, max_ratio = self.config.aspect_ratio_range

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            rect_area = float(w * h)
            if rect_area <= 0 or rect_area > max_area:
                continue

            solidity = area / rect_area
            if solidity < self.config.min_solidity:
                continue

            if h == 0 or w == 0:
                continue
            ratio = max(w, h) / float(min(w, h))
            if ratio < min_ratio or ratio > max_ratio:
                continue

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
            if len(approx) < 4 or len(approx) > 8:
                continue

            pad_x = int(round(w * self.config.padding_ratio))
            pad_y = int(round(h * self.config.padding_ratio))
            x0 = max(x - pad_x, 0)
            y0 = max(y - pad_y, 0)
            x1 = min(x + w + pad_x, width)
            y1 = min(y + h + pad_y, height)
            candidates.append((x0, y0, x1 - x0, y1 - y0))

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

        return []


__all__ = ["LicensePlateDetector"]
