"""Text recognition utilities for license plate crops."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

try:  # Optional dependency used by the deep recognizer
    import onnxruntime as ort
except ImportError:  # pragma: no cover - optional dependency
    ort = None


@dataclass
class PlateReading:
    """Container holding recognition details for a single plate."""

    bbox: Tuple[int, int, int, int]
    text: str
    confidence: float


class CharacterTemplateLibrary:
    """Creates binary templates for alphanumeric characters."""

    def __init__(
        self,
        *,
        characters: str | None = None,
        font: int = cv2.FONT_HERSHEY_SIMPLEX,
        font_scale: float = 1.3,
        thickness: int = 3,
        template_size: Tuple[int, int] = (60, 40),
    ) -> None:
        self.characters = characters or "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.font = font
        self.font_scale = font_scale
        self.thickness = thickness
        self.template_size = template_size
        self.templates = self._build_templates()

    def _build_templates(self) -> dict[str, np.ndarray]:
        templates: dict[str, np.ndarray] = {}
        height, width = self.template_size
        baseline = 0
        for char in self.characters:
            image = np.zeros((height, width), dtype=np.uint8)
            text_size, baseline = cv2.getTextSize(
                char, self.font, self.font_scale, self.thickness
            )
            text_width, text_height = text_size
            baseline = int(baseline)
            origin_x = max((width - text_width) // 2, 0)
            origin_y = min(height - baseline, (height + text_height) // 2)
            cv2.putText(
                image,
                char,
                (origin_x, origin_y),
                self.font,
                self.font_scale,
                255,
                self.thickness,
            )
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
            templates[char] = binary
        return templates

    def match(self, glyph: np.ndarray) -> Tuple[str, float]:
        glyph = cv2.resize(glyph, (self.template_size[1], self.template_size[0]))
        glyph = glyph.astype(np.float32) / 255.0
        best_char = ""
        best_score = -1.0
        for char, template in self.templates.items():
            template_float = template.astype(np.float32) / 255.0
            numerator = float(np.sum(glyph * template_float))
            denominator = float(
                np.sqrt(np.sum(glyph**2) * np.sum(template_float**2))
            )
            if denominator == 0:
                continue
            score = numerator / denominator
            if score > best_score:
                best_score = score
                best_char = char
        return best_char, max(best_score, 0.0)


class LicensePlateRecognizer:
    """Recognizes alphanumeric content from detected license plate regions."""

    def __init__(
        self,
        *,
        templates: CharacterTemplateLibrary | None = None,
        min_character_score: float = 0.5,
    ) -> None:
        self.templates = templates or CharacterTemplateLibrary()
        self.min_character_score = float(min_character_score)

    def _prepare_binary(self, crop: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        blurred = cv2.GaussianBlur(normalized, (5, 5), 0)
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        if np.mean(binary) > 127:
            binary = 255 - binary
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        return binary

    def _extract_glyphs(
        self, binary: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        height, width = binary.shape[:2]
        glyph_boxes: List[Tuple[int, int, int, int]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h < 0.01 * width * height:
                continue
            if h < 0.45 * height or h > 0.95 * height:
                continue
            if w < 0.03 * width or w > 0.35 * width:
                continue
            glyph_boxes.append((x, y, w, h))
        glyph_boxes.sort(key=lambda box: box[0])
        return glyph_boxes

    def recognize(
        self, frame: np.ndarray, boxes: Iterable[Tuple[int, int, int, int]]
    ) -> List[PlateReading]:
        """Return best-effort text predictions for the provided bounding boxes."""

        readings: List[PlateReading] = []
        for (x, y, w, h) in boxes:
            crop = frame[y : y + h, x : x + w]
            if crop.size == 0:
                continue

            binary = self._prepare_binary(crop)
            glyph_boxes = self._extract_glyphs(binary)
            if not glyph_boxes:
                continue

            characters: List[str] = []
            scores: List[float] = []
            for gx, gy, gw, gh in glyph_boxes:
                glyph = binary[gy : gy + gh, gx : gx + gw]
                char, score = self.templates.match(glyph)
                if not char or score < self.min_character_score:
                    continue
                characters.append(char)
                scores.append(score)

            if not characters:
                continue

            text = "".join(characters)
            confidence = float(np.mean(scores))
            readings.append(PlateReading((x, y, w, h), text, confidence))

        return readings


@dataclass
class CrnnRecognizerConfig:
    """Configuration for :class:`CrnnPlateRecognizer`."""

    model_path: str
    alphabet: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    input_size: Tuple[int, int] = (32, 160)
    mean: float = 0.5
    std: float = 0.5
    blank_index: int | None = None
    providers: Sequence[str] | None = None
    min_text_confidence: float = 0.5


class CrnnPlateRecognizer:
    """CTC-based recognizer backed by an ONNX CRNN/LPRNet model."""

    def __init__(self, config: CrnnRecognizerConfig) -> None:
        if ort is None:  # pragma: no cover - exercised when dependency missing
            raise ImportError(
                "onnxruntime is required to use CrnnPlateRecognizer. Install it "
                "with `pip install onnxruntime`."
            )

        self.config = config
        self.model_path = Path(config.model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"CRNN model not found at {self.model_path!s}. Provide an ONNX "
                "export trained for license-plate transcription."
            )

        providers = config.providers or ort.get_available_providers()
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_height, self.input_width = config.input_size
        if self.input_height <= 0 or self.input_width <= 0:
            raise ValueError("input_size must contain positive integers")

        self.alphabet = config.alphabet
        if not self.alphabet:
            raise ValueError("alphabet must contain at least one character")
        self.blank_index = (
            config.blank_index if config.blank_index is not None else len(self.alphabet)
        )

    def _prepare(self, crop: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(
            gray, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR
        )
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - self.config.mean) / max(self.config.std, 1e-6)
        tensor = normalized[np.newaxis, np.newaxis, :, :]
        return tensor.astype(np.float32)

    @staticmethod
    def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
        logits = logits - np.max(logits, axis=axis, keepdims=True)
        exp = np.exp(logits)
        return exp / np.sum(exp, axis=axis, keepdims=True)

    def _decode(self, probabilities: np.ndarray) -> Tuple[str, float]:
        indices = np.argmax(probabilities, axis=1)
        confidence_scores = probabilities[np.arange(probabilities.shape[0]), indices]
        characters: List[str] = []
        scores: List[float] = []
        prev_index = -1
        for index, score in zip(indices, confidence_scores):
            if index == self.blank_index:
                prev_index = index
                continue
            if index == prev_index:
                prev_index = index
                continue
            if index < 0 or index >= len(self.alphabet):
                prev_index = index
                continue
            characters.append(self.alphabet[index])
            scores.append(float(score))
            prev_index = index

        if not characters:
            return "", 0.0

        confidence = float(np.mean(scores)) if scores else 0.0
        return "".join(characters), confidence

    def recognize(
        self, frame: np.ndarray, boxes: Iterable[Tuple[int, int, int, int]]
    ) -> List[PlateReading]:
        readings: List[PlateReading] = []
        for (x, y, w, h) in boxes:
            crop = frame[y : y + h, x : x + w]
            if crop.size == 0:
                continue

            tensor = self._prepare(crop)
            outputs = self.session.run(None, {self.input_name: tensor})
            logits = outputs[0]
            if logits.ndim == 3:
                if logits.shape[0] == 1:
                    logits = logits[0]
                elif logits.shape[1] == 1:
                    logits = np.squeeze(logits, axis=1)
                else:
                    logits = logits[0]
            logits = np.squeeze(logits, axis=0) if logits.ndim == 3 else logits
            if logits.ndim != 2:
                raise ValueError(
                    "Unexpected CRNN output shape; expected (time, classes)."
                )
            if logits.shape[0] < logits.shape[1]:
                probabilities = self._softmax(logits, axis=1)
            else:
                probabilities = self._softmax(logits.T, axis=1)
            text, confidence = self._decode(probabilities)
            if not text:
                continue
            if confidence < self.config.min_text_confidence:
                continue
            readings.append(PlateReading((x, y, w, h), text, confidence))

        return readings


__all__ = [
    "LicensePlateRecognizer",
    "PlateReading",
    "CharacterTemplateLibrary",
    "CrnnRecognizerConfig",
    "CrnnPlateRecognizer",
]
