"""
Face package initialization.

This package provides comprehensive face detection and recognition capabilities.
"""

from .detector import FaceDetector
from .recognizer import FaceRecognizer
from .preprocessing import EnhancedFacePreprocessor as FacePreprocessor
from .antispoofing import (
    AntiSpoofingDetector,
    DetectionMode,
    LivenessLevel,
    SpoofingAttackType,
    create_basic_detector,
    create_high_security_detector,
    create_video_detector,
    create_lightweight_detector
)
from .antispoofing_integration import (
    PerfectAntiSpoof,
    SecureFaceRecognition,
    SecurityLevel,
    AttackType,
    AntiSpoofResult
)

__all__ = [
    "FaceDetector",
    "FaceRecognizer", 
    "FacePreprocessor",
    "AntiSpoofingDetector",
    "DetectionMode",
    "LivenessLevel",
    "SpoofingAttackType",
    "create_basic_detector",
    "create_high_security_detector",
    "create_video_detector",
    "create_lightweight_detector",
    "PerfectAntiSpoof",
    "SecureFaceRecognition",
    "SecurityLevel",
    "AttackType",
    "AntiSpoofResult",
]


