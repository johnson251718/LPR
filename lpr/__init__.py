"""License plate recognition package."""

from .detection import (
    ContourDetectorConfig,
    ContourLicensePlateDetector,
    LicensePlateDetector,
    YoloDetectorConfig,
    YoloLicensePlateDetector,
)
from .pipeline import PlateRecognitionPipeline
from .recognition import (
    CharacterTemplateLibrary,
    CrnnPlateRecognizer,
    CrnnRecognizerConfig,
    LicensePlateRecognizer,
)

__all__ = [
    "ContourDetectorConfig",
    "ContourLicensePlateDetector",
    "LicensePlateDetector",
    "YoloDetectorConfig",
    "YoloLicensePlateDetector",
    "CharacterTemplateLibrary",
    "CrnnRecognizerConfig",
    "CrnnPlateRecognizer",
    "LicensePlateRecognizer",
    "PlateRecognitionPipeline",
]
