"""
Face package initialization.

This package provides comprehensive face detection and recognition capabilities.
"""

from .detector import FaceDetector
from .recognizer import FaceRecognizer
from .preprocessing import EnhancedFacePreprocessor as FacePreprocessor
from .final_anti_spoof import (
    UltimateAntiSpoof,
    SecurityLevel,
    AttackType,
    AntiSpoofResult
)

__all__ = [
    "FaceDetector",
    "FaceRecognizer", 
    "FacePreprocessor",
    "UltimateAntiSpoof",
    "SecurityLevel",
    "AttackType",
    "AntiSpoofResult",
]


