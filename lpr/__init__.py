"""License plate recognition package."""

from .detection import DetectorConfig, LicensePlateDetector
from .recognition import CharacterTemplateLibrary, LicensePlateRecognizer
from .pipeline import PlateRecognitionPipeline

__all__ = [
    "DetectorConfig",
    "LicensePlateDetector",
    "CharacterTemplateLibrary",
    "LicensePlateRecognizer",
    "PlateRecognitionPipeline",
]
