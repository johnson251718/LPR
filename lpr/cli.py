"""Command line interface for the license plate recognition pipeline."""

from __future__ import annotations

import argparse
from typing import Sequence

from .detection import (
    ContourDetectorConfig,
    ContourLicensePlateDetector,
    YoloDetectorConfig,
    YoloLicensePlateDetector,
)
from .pipeline import PlateRecognitionPipeline
from .recognition import CrnnPlateRecognizer, CrnnRecognizerConfig, LicensePlateRecognizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="License plate recognition demo")
    parser.add_argument(
        "--detector",
        choices=("contour", "yolo"),
        default="contour",
        help="Detector backend to use (default: contour).",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=5,
        help="Maximum number of candidate plates retained per frame (default: 5)",
    )
    parser.add_argument(
        "--yolo-model",
        help="Path to an ONNX YOLO license-plate detector (required for YOLO mode)",
    )
    parser.add_argument(
        "--yolo-input-width",
        type=int,
        default=640,
        help="Input width for the YOLO detector (default: 640)",
    )
    parser.add_argument(
        "--yolo-input-height",
        type=int,
        default=640,
        help="Input height for the YOLO detector (default: 640)",
    )
    parser.add_argument(
        "--yolo-confidence",
        type=float,
        default=0.25,
        help="Minimum confidence threshold for YOLO detections (default: 0.25)",
    )
    parser.add_argument(
        "--yolo-nms",
        type=float,
        default=0.45,
        help="NMS IoU threshold for YOLO detections (default: 0.45)",
    )
    parser.add_argument(
        "--yolo-class-id",
        type=int,
        action="append",
        help="Restrict YOLO detections to specific class IDs (repeatable)",
    )

    parser.add_argument(
        "--recognizer",
        choices=("template", "crnn"),
        default="template",
        help="Text recognition backend (default: template)",
    )
    parser.add_argument(
        "--min-character-score",
        type=float,
        default=0.5,
        help=(
            "Minimum normalized correlation score required to accept a character "
            "match in template mode (default: 0.5)"
        ),
    )
    parser.add_argument(
        "--crnn-model",
        help="Path to an ONNX CRNN/LPRNet recognizer (required for CRNN mode)",
    )
    parser.add_argument(
        "--crnn-alphabet",
        default="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        help="Alphabet used by the CRNN model (default: digits+uppercase letters)",
    )
    parser.add_argument(
        "--crnn-input-width",
        type=int,
        default=160,
        help="Input width for the CRNN recognizer (default: 160)",
    )
    parser.add_argument(
        "--crnn-input-height",
        type=int,
        default=32,
        help="Input height for the CRNN recognizer (default: 32)",
    )
    parser.add_argument(
        "--crnn-mean",
        type=float,
        default=0.5,
        help="Mean normalization value for CRNN preprocessing (default: 0.5)",
    )
    parser.add_argument(
        "--crnn-std",
        type=float,
        default=0.5,
        help="Standard deviation for CRNN preprocessing (default: 0.5)",
    )
    parser.add_argument(
        "--crnn-blank-index",
        type=int,
        help="Blank index for CTC decoding (default: len(alphabet))",
    )
    parser.add_argument(
        "--crnn-min-confidence",
        type=float,
        default=0.5,
        help="Minimum mean character confidence accepted in CRNN mode (default: 0.5)",
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", help="Run recognition on a still image")
    input_group.add_argument("--video", help="Run recognition on a video file")

    parser.add_argument(
        "--output",
        help="Optional path where an annotated image/video will be saved",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Limit the number of processed frames when running on video",
    )

    return parser


def main(args: Sequence[str] | None = None) -> int:
    parser = build_parser()
    namespace = parser.parse_args(args=args)

    if namespace.detector == "yolo":
        if not namespace.yolo_model:
            parser.error("--yolo-model is required when --detector=yolo")
        yolo_config = YoloDetectorConfig(
            model_path=namespace.yolo_model,
            input_size=(namespace.yolo_input_width, namespace.yolo_input_height),
            confidence_threshold=namespace.yolo_confidence,
            nms_threshold=namespace.yolo_nms,
            max_candidates=namespace.max_candidates,
            class_ids=tuple(namespace.yolo_class_id) if namespace.yolo_class_id else None,
        )
        detector = YoloLicensePlateDetector(yolo_config)
    else:
        contour_config = ContourDetectorConfig(max_candidates=namespace.max_candidates)
        detector = ContourLicensePlateDetector(contour_config)

    if namespace.recognizer == "crnn":
        if not namespace.crnn_model:
            parser.error("--crnn-model is required when --recognizer=crnn")
        crnn_config = CrnnRecognizerConfig(
            model_path=namespace.crnn_model,
            alphabet=namespace.crnn_alphabet,
            input_size=(namespace.crnn_input_height, namespace.crnn_input_width),
            mean=namespace.crnn_mean,
            std=namespace.crnn_std,
            blank_index=namespace.crnn_blank_index,
            min_text_confidence=namespace.crnn_min_confidence,
        )
        recognizer = CrnnPlateRecognizer(crnn_config)
    else:
        recognizer = LicensePlateRecognizer(
            min_character_score=namespace.min_character_score
        )
    pipeline = PlateRecognitionPipeline(detector=detector, recognizer=recognizer)

    if namespace.image:
        try:
            readings = pipeline.process_image(
                namespace.image, output_path=namespace.output
            )
        except FileNotFoundError as error:
            parser.error(str(error))
        if readings:
            for reading in readings:
                print(
                    f"Plate {reading.text} detected with confidence "
                    f"{reading.confidence:.2f}"
                )
        else:
            print("No plates detected")
        return 0

    assert namespace.video
    try:
        results = pipeline.process_video(
            namespace.video,
            output_path=namespace.output,
            max_frames=namespace.max_frames,
        )
    except FileNotFoundError as error:
        parser.error(str(error))
    if not results:
        print("No frames processed")
        return 0

    total_detections = sum(len(frame.readings) for frame in results)
    print(f"Processed {len(results)} frames with {total_detections} total detections")
    for frame in results:
        for reading in frame.readings:
            print(
                f"Frame {frame.frame_index}: {reading.text} "
                f"(confidence {reading.confidence:.2f})"
            )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
