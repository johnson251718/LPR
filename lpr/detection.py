"""Utilities for locating potential license plates in images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class ContourDetectorConfig:
    """Configuration values for :class:`ContourLicensePlateDetector`."""

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


class ContourLicensePlateDetector:
    """Detects potential license plate regions using contour analysis."""

    def __init__(self, config: ContourDetectorConfig | None = None) -> None:
        self.config = config or ContourDetectorConfig()

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


@dataclass
class YoloDetectorConfig:
    """Configuration for :class:`YoloLicensePlateDetector`."""

    model_path: str
    input_size: Tuple[int, int] = (640, 640)
    confidence_threshold: float = 0.25
    nms_threshold: float = 0.45
    max_candidates: int = 10
    class_ids: Sequence[int] | None = None


class YoloLicensePlateDetector:
    """YOLO-based detector implemented with :mod:`cv2.dnn`.

    The detector expects an ONNX export of a YOLO model that predicts license plates.
    Only the ONNX runtime built into OpenCV is required—no Ultralytics runtime is used.
    """

    def __init__(self, config: YoloDetectorConfig) -> None:
        self.config = config
        self.model_path = Path(config.model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YOLO model not found at {self.model_path!s}. Download an ONNX "
                "export of a license-plate detector and provide its path via the "
                "configuration."
            )
        self.net = cv2.dnn.readNetFromONNX(str(self.model_path))
        width, height = self.config.input_size
        if width <= 0 or height <= 0:
            raise ValueError("input_size must contain positive integers")
        self.input_width = int(width)
        self.input_height = int(height)

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if frame is None or frame.size == 0:
            raise ValueError("frame must be a non-empty numpy array")

    @staticmethod
    def _letterbox(
        image: np.ndarray, new_width: int, new_height: int
    ) -> Tuple[np.ndarray, float, float, float]:
        height, width = image.shape[:2]
        scale = min(new_width / width, new_height / height)
        scaled_width = int(round(width * scale))
        scaled_height = int(round(height * scale))
        resized = cv2.resize(image, (scaled_width, scaled_height), cv2.INTER_LINEAR)
        canvas = np.full((new_height, new_width, 3), 114, dtype=np.uint8)
        dw = (new_width - scaled_width) / 2.0
        dh = (new_height - scaled_height) / 2.0
        left = int(round(dw - 0.1))
        right = int(round(dw + scaled_width + 0.1))
        top = int(round(dh - 0.1))
        bottom = int(round(dh + scaled_height + 0.1))
        left = max(left, 0)
        top = max(top, 0)
        right = min(right, new_width)
        bottom = min(bottom, new_height)
        canvas[top:bottom, left:right] = resized
        return canvas, scale, float(left), float(top)

    def _decode_predictions(
        self,
        predictions: np.ndarray,
        scale: float,
        dw: float,
        dh: float,
        frame_shape: Tuple[int, int, int],
    ) -> Tuple[List[List[int]], List[float]]:
        predictions = predictions.reshape(-1, predictions.shape[-1])
        frame_height, frame_width = frame_shape[:2]
        boxes: List[List[int]] = []
        scores: List[float] = []
        allowed_classes = set(self.config.class_ids) if self.config.class_ids else None

        for pred in predictions:
            if pred.shape[0] < 6:
                continue
            objectness = float(pred[4])
            if objectness < 1e-6:
                continue
            class_scores = pred[5:]
            if class_scores.size == 0:
                continue
            class_id = int(np.argmax(class_scores))
            if allowed_classes is not None and class_id not in allowed_classes:
                continue
            class_score = float(class_scores[class_id])
            confidence = objectness * class_score
            if confidence < self.config.confidence_threshold:
                continue

            cx, cy, width, height = pred[:4].astype(float)
            x = (cx - width / 2.0 - dw) / scale
            y = (cy - height / 2.0 - dh) / scale
            w = width / scale
            h = height / scale

            x = max(0.0, min(x, frame_width - 1.0))
            y = max(0.0, min(y, frame_height - 1.0))
            w = max(1.0, min(w, frame_width - x))
            h = max(1.0, min(h, frame_height - y))

            boxes.append([int(round(x)), int(round(y)), int(round(w)), int(round(h))])
            scores.append(confidence)

        return boxes, scores

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Return bounding boxes for detected license plates."""

        self._validate_frame(frame)

        letterboxed, scale, pad_x, pad_y = self._letterbox(
            frame, self.input_width, self.input_height
        )
        blob = cv2.dnn.blobFromImage(
            letterboxed,
            scalefactor=1.0 / 255.0,
            size=(self.input_width, self.input_height),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        outputs = self.net.forward()
        if isinstance(outputs, (list, tuple)):
            predictions = outputs[0]
        else:
            predictions = outputs

        if predictions.ndim == 3 and predictions.shape[0] == 1:
            predictions = predictions[0]

        boxes, scores = self._decode_predictions(
            predictions, scale, pad_x, pad_y, frame.shape
        )
        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            score_threshold=self.config.confidence_threshold,
            nms_threshold=self.config.nms_threshold,
        )

        if isinstance(indices, tuple):
            indices = indices[0]

        selected: List[Tuple[int, int, int, int]] = []
        for idx in np.atleast_1d(indices).flatten().tolist():
            if idx < 0 or idx >= len(boxes):
                continue
            selected.append(tuple(boxes[idx]))
            if (
                self.config.max_candidates
                and len(selected) >= self.config.max_candidates
            ):
                break

        return selected


# Maintain backwards compatibility with the previous API name.
LicensePlateDetector = ContourLicensePlateDetector


__all__ = [
    "ContourDetectorConfig",
    "ContourLicensePlateDetector",
    "YoloDetectorConfig",
    "YoloLicensePlateDetector",
    "LicensePlateDetector",
]
