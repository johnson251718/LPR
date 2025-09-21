"""Command line interface for the license plate recognition pipeline."""

from __future__ import annotations

import argparse
from typing import Sequence

from .detection import LicensePlateDetector
from .pipeline import PlateRecognitionPipeline
from .recognition import LicensePlateRecognizer


def _parse_languages(value: str) -> Sequence[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="License plate recognition demo")
    parser.add_argument(
        "--cascade-path",
        help=(
            "Path to the Haar cascade XML. Defaults to resources/haarcascade_russian_"
            "plate_number.xml"
        ),
    )
    parser.add_argument(
        "--languages",
        default="en",
        help="Comma separated list of EasyOCR languages (default: en)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable GPU acceleration in EasyOCR if supported",
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

    try:
        detector = LicensePlateDetector(cascade_path=namespace.cascade_path)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    recognizer = LicensePlateRecognizer(
        languages=_parse_languages(namespace.languages), gpu=namespace.gpu
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
