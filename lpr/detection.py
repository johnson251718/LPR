"""YOLO-based license plate detection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, TYPE_CHECKING

try:  # pragma: no cover - optional dependency
    import numpy as np
except Exception:  # pragma: no cover - numpy may be unavailable in minimal envs
    np = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import cv2  # type: ignore
except Exception:  # pragma: no cover - fallback if OpenCV is unavailable
    cv2 = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import onnxruntime as ort
except Exception:  # pragma: no cover - fallback if onnxruntime is unavailable
    ort = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from onnxruntime import InferenceSession


@dataclass
class Detection:
    """Container describing a single detection result."""

    box: Tuple[int, int, int, int]
    score: float
    class_id: int


class YoloLicensePlateDetector:
    """License plate detector powered by YOLO exported to ONNX.

    The detector keeps the preprocessing steps used by the original OpenCV
    implementation (letterboxing and blob scaling) while delegating model
    execution to :mod:`onnxruntime`.
    """

    def __init__(
        self,
        model_path: str,
        *,
        input_size: Tuple[int, int] = (640, 640),
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        allowed_classes: Optional[Sequence[int]] = None,
        providers: Optional[Sequence[str]] = None,
        session: Optional["InferenceSession"] = None,
    ) -> None:
        """Initialise the detector.

        Parameters
        ----------
        model_path:
            Path to the ONNX model file.
        input_size:
            Model input size as ``(width, height)``.
        conf_threshold:
            Minimum confidence required for a detection to be kept.
        iou_threshold:
            Maximum allowed IoU during non-maximum suppression.
        allowed_classes:
            Optional iterable restricting detections to specific class IDs.
        providers:
            Execution providers passed to :class:`onnxruntime.InferenceSession`.
        session:
            Pre-built inference session. When provided, the detector will use it
            instead of creating a new session from ``model_path``.
        """

        if session is not None:
            self.session = session
        else:
            if ort is None:  # pragma: no cover - defensive branch
                raise ImportError(
                    "onnxruntime is required to load the YOLO detector. Install "
                    "the 'onnxruntime' or 'onnxruntime-gpu' package to proceed."
                )

            available = ort.get_available_providers()
            if providers is None:
                selected_providers = available
            else:
                selected_providers = list(providers)
                unsupported = set(selected_providers) - set(available)
                if unsupported:  # pragma: no cover - defensive branch
                    raise ValueError(
                        "The following onnxruntime providers are not available: "
                        f"{', '.join(sorted(unsupported))}"
                    )

            session_options = ort.SessionOptions()
            self.session = ort.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=selected_providers,
            )

        self.input_size = input_size
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.allowed_classes = (
            set(int(cls) for cls in allowed_classes)
            if allowed_classes is not None
            else None
        )

        inputs = getattr(self.session, "get_inputs", lambda: [])()
        if not inputs:  # pragma: no cover - defensive branch
            raise ValueError("The provided ONNX session does not expose any inputs.")
        self.input_name = inputs[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]

    # ------------------------------------------------------------------
    # Pre-processing utilities
    # ------------------------------------------------------------------
    def _letterbox(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """Resize ``image`` while keeping its aspect ratio via letterboxing."""

        if np is None:
            raise ImportError(
                "numpy is required for image preprocessing but is not installed."
            )

        if image.ndim != 3:
            raise ValueError("Expected a 3D image tensor in HWC format.")

        orig_h, orig_w = image.shape[:2]
        target_w, target_h = self.input_size

        if orig_w == 0 or orig_h == 0:
            raise ValueError("Input image must have non-zero dimensions.")

        scale = min(target_w / orig_w, target_h / orig_h)
        resize_w = max(int(round(orig_w * scale)), 1)
        resize_h = max(int(round(orig_h * scale)), 1)

        resized = self._resize(image, (resize_w, resize_h))

        pad_w = target_w - resize_w
        pad_h = target_h - resize_h
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        if cv2 is not None:  # pragma: no branch - prefer OpenCV when available
            padded = cv2.copyMakeBorder(
                resized,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_CONSTANT,
                value=(114, 114, 114),
            )
        else:
            padded = np.full(
                (target_h, target_w, image.shape[2]),
                114,
                dtype=image.dtype,
            )
            padded[pad_top : pad_top + resize_h, pad_left : pad_left + resize_w] = resized

        return padded, scale, (float(pad_left), float(pad_top))

    def _resize(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Resize ``image`` to ``size`` using either OpenCV or nearest-neighbour."""

        if np is None:
            raise ImportError(
                "numpy is required for image resizing but is not installed."
            )

        width, height = size
        if cv2 is not None:  # pragma: no branch - prefer OpenCV when available
            return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

        orig_h, orig_w = image.shape[:2]
        if orig_h == height and orig_w == width:
            return image.copy()

        y_indices = (np.floor(np.arange(height) * orig_h / height)).astype(int)
        x_indices = (np.floor(np.arange(width) * orig_w / width)).astype(int)
        y_indices = np.clip(y_indices, 0, orig_h - 1)
        x_indices = np.clip(x_indices, 0, orig_w - 1)
        return image[y_indices[:, None], x_indices]

    # ------------------------------------------------------------------
    # Post-processing utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _squeeze_singleton(data):
        """Remove leading singleton dimensions from nested sequences."""

        result = data
        while isinstance(result, (list, tuple)) and len(result) == 1:
            result = result[0]
        return result

    def _prepare_prediction_rows(self, predictions) -> List[List[float]]:
        """Convert predictions to a 2D list of floats."""

        if np is not None and isinstance(predictions, np.ndarray):
            array = np.squeeze(predictions)
            if array.ndim == 0:
                return []
            if array.ndim == 1:
                return [[float(value) for value in array.tolist()]]
            if array.ndim == 2:
                return [[float(value) for value in row] for row in array.tolist()]
            raise ValueError(
                "Predictions must be 2-dimensional after removing singleton axes."
            )

        if hasattr(predictions, "tolist") and np is None:
            try:
                predictions = predictions.tolist()
            except Exception:  # pragma: no cover - defensive fallback
                pass

        predictions = self._squeeze_singleton(predictions)
        if isinstance(predictions, (list, tuple)):
            if not predictions:
                return []
            first = predictions[0]
            if isinstance(first, (list, tuple)):
                return [
                    [float(value) for value in row]
                    for row in predictions
                ]
            if isinstance(first, (int, float)):
                return [[float(value) for value in predictions]]

        raise ValueError("Predictions must be convertible to a 2D array-like structure.")

    @staticmethod
    def _transpose(matrix: List[List[float]]) -> List[List[float]]:
        if not matrix:
            return []
        return [list(row) for row in zip(*matrix)]

    def _decode_predictions(
        self,
        predictions,
        original_shape: Tuple[int, int],
        scale: float,
        pad: Tuple[float, float],
    ) -> List[Detection]:
        """Transform raw model ``predictions`` into usable detections."""

        rows = self._prepare_prediction_rows(predictions)
        if not rows:
            return []

        if rows and len(rows[0]) < 5:
            rows = self._transpose(rows)
        elif len(rows) >= 5 and len(rows) > len(rows[0]):
            rows = self._transpose(rows)

        if not rows:
            return []

        attribute_count = len(rows[0])
        for row in rows:
            if len(row) != attribute_count:
                raise ValueError("Inconsistent attribute counts in prediction rows.")

        if attribute_count < 5:
            raise ValueError("Predictions do not contain enough attributes per box.")

        boxes: List[List[float]] = []
        scores: List[float] = []
        class_ids: List[int] = []

        for row in rows:
            x, y, w, h = (float(value) for value in row[:4])
            objectness = float(row[4])
            class_scores = [float(value) for value in row[5:]]

            if class_scores:
                best_idx = max(range(len(class_scores)), key=class_scores.__getitem__)
                best_score = class_scores[best_idx]
                confidence = objectness * best_score
                cls_id = best_idx
            else:
                confidence = objectness
                cls_id = 0

            if confidence < self.conf_threshold:
                continue
            if self.allowed_classes is not None and cls_id not in self.allowed_classes:
                continue

            boxes.append([x, y, w, h])
            scores.append(confidence)
            class_ids.append(cls_id)

        if not boxes:
            return []

        boxes_xyxy = [self._xywh_to_xyxy_list(box) for box in boxes]
        left_pad, top_pad = pad
        scale = max(scale, 1e-6)
        adjusted_boxes: List[List[float]] = []
        for box in boxes_xyxy:
            x1 = (box[0] - left_pad) / scale
            y1 = (box[1] - top_pad) / scale
            x2 = (box[2] - left_pad) / scale
            y2 = (box[3] - top_pad) / scale
            adjusted_boxes.append([x1, y1, x2, y2])

        height, width = original_shape
        for box in adjusted_boxes:
            box[0] = min(max(box[0], 0.0), float(width))
            box[1] = min(max(box[1], 0.0), float(height))
            box[2] = min(max(box[2], 0.0), float(width))
            box[3] = min(max(box[3], 0.0), float(height))

        keep = self._nms(adjusted_boxes, scores)
        detections: List[Detection] = []
        for idx in keep:
            x1, y1, x2, y2 = adjusted_boxes[idx]
            detections.append(
                Detection(
                    box=(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
                    score=float(scores[idx]),
                    class_id=int(class_ids[idx]),
                )
            )
        return detections

    @staticmethod
    def _xywh_to_xyxy_list(box: Sequence[float]) -> List[float]:
        """Convert a single ``(x, y, w, h)`` box to ``(x1, y1, x2, y2)`` format."""

        x, y, w, h = box
        half_w = w / 2.0
        half_h = h / 2.0
        return [x - half_w, y - half_h, x + half_w, y + half_h]

    def _nms(self, boxes: List[List[float]], scores: Sequence[float]) -> List[int]:
        """Perform Non-Maximum Suppression (NMS)."""

        if not boxes:
            return []

        order = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
        keep: List[int] = []

        while order:
            current = order.pop(0)
            keep.append(current)

            remaining: List[int] = []
            for idx in order:
                iou = self._iou(boxes[current], boxes[idx])
                if iou <= self.iou_threshold:
                    remaining.append(idx)
            order = remaining

        return keep

    @staticmethod
    def _iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
        """Compute the Intersection-over-Union of two boxes."""

        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter_area
        if denom <= 0:
            return 0.0
        return inter_area / denom

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(self, image: np.ndarray) -> List[Detection]:
        """Detect license plates in ``image``.

        Parameters
        ----------
        image:
            Input image in ``HWC`` format. The channel order is expected to be the
            same one the ONNX model was trained on (typically BGR).
        """

        if np is None:
            raise ImportError(
                "numpy is required for model preprocessing but is not installed."
            )

        processed, scale, pad = self._letterbox(image)
        blob = processed.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]

        outputs = self.session.run(self.output_names, {self.input_name: blob})
        return self._decode_predictions(outputs[0], image.shape[:2], scale, pad)


__all__ = ["Detection", "YoloLicensePlateDetector"]
