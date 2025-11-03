"""Face recognition module using LBPH algorithm.

Implements the FaceRecognizer class for identifying individuals from detected face regions.
Uses Local Binary Pattern Histogram (LBPH) algorithm for robust face recognition.
"""

from __future__ import annotations

import os
import pickle
import time
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .preprocessing import EnhancedFacePreprocessor as FacePreprocessor
from .final_anti_spoof import (
    UltimateAntiSpoof,
    SecurityLevel,
    AntiSpoofResult,
    AttackType
)


class LightingCondition(Enum):
    """Lighting condition categories."""
    DIM = "dim"
    NORMAL = "normal"
    BRIGHT = "bright"
    UNKNOWN = "unknown"


@dataclass
class LightingMetrics:
    """Container for lighting assessment metrics."""
    condition: LightingCondition
    brightness: float
    contrast: float
    quality: float


@dataclass
class RecognitionResult:
    """Container for recognition results with all metadata."""
    recognized: bool
    label: int
    name: str
    confidence: float
    raw_confidence: float
    threshold: float
    quality_score: float
    lighting_metrics: LightingMetrics
    liveness_result: Optional[AntiSpoofResult] = None  # Anti-spoofing result from UltimateAntiSpoof
    is_live: bool = True  # Liveness status
    
    def __str__(self) -> str:
        """Human-readable representation."""
        status = "✓ Recognized" if self.recognized else "✗ Not Recognized"
        liveness_status = "✓ LIVE" if self.is_live else "✗ SPOOF"
        return (
            f"{status}: {self.name} "
            f"(conf: {self.confidence:.1f}, quality: {self.quality_score:.2f}, {liveness_status})"
        )


class FaceRecognizer:
    """
    Face recognition system using Local Binary Pattern Histogram (LBPH) algorithm.
    
    This class provides comprehensive face recognition capabilities including training,
    prediction, and model management for attendance systems with advanced features:
    
    - Adaptive lighting normalization and threshold adjustment
    - Multi-stage quality assessment
    - Temporal confidence smoothing with outlier rejection
    - Label consistency tracking
    - Configurable recognition strictness
    
    Example:
        >>> recognizer = FaceRecognizer(threshold=50.0)
        >>> recognizer.train(face_samples, labels, label_names)
        >>> result = recognizer.predict_with_name(face_image)
        >>> if result.recognized:
        ...     print(f"Recognized: {result.name}")
    """

    # Class constants
    DEFAULT_FACE_SIZE = (150, 150)  # Match capture_training_data.py resolution
    MIN_TRAINING_SAMPLES = 1
    MAX_CONFIDENCE_HISTORY = 10
    MAX_QUALITY_HISTORY = 5
    MIN_SAMPLES_FOR_RECOGNITION = 5  # Require 5 consecutive frames before recognizing
    CONFIDENCE_STABILITY_THRESHOLD = 15.0
    PERSON_SWITCH_THRESHOLD = 15.0  # New person must be 15 points better to switch
    
    # Quality thresholds - WEBCAM OPTIMIZED
    MIN_QUALITY_THRESHOLD = 0.30  # Lowered for webcam quality
    HIGH_QUALITY_THRESHOLD = 0.75  # Lowered for webcam quality
    LOW_QUALITY_THRESHOLD = 0.40  # Lowered for webcam quality
    
    # Confidence bounds - WEBCAM OPTIMIZED
    ULTRA_STRICT_THRESHOLD = 120.0  # More lenient
    UNCERTAIN_RANGE = (110.0, 130.0)  # More lenient range
    MAX_CONFIDENCE_THRESHOLD = 140.0  # More lenient threshold
    
    # Blink verification constants (based on research) - LENIENT DEFAULTS
    MIN_BLINK_INTERVAL = 0.5  # Minimum seconds between blinks (very lenient)
    MAX_BLINK_INTERVAL = 20.0  # Maximum seconds without blink (very lenient)
    REQUIRED_BLINKS_FOR_VERIFICATION = 1  # Just 1 blink needed (very lenient)
    BLINK_VERIFICATION_WINDOW = 30.0  # Longer time window (30 seconds)

    def __init__(
        self,
        radius: int = 1,  # Keep original for stability
        neighbors: int = 8,  # Keep original for stability
        grid_x: int = 8,  # Keep original for stability
        grid_y: int = 8,  # Keep original for stability
        threshold: float = 50.0,
        use_enhanced_preprocessing: bool = True,
        enable_antispoofing: bool = False,
        antispoofing_mode: str = "basic",  # "basic", "high_security", "disabled"
        reject_on_spoof: bool = True,
        enable_blink_verification: bool = False,  # Disabled by default (experimental)
        min_blink_interval: float = 0.5,  # Very lenient - minimum seconds between blinks
        max_blink_interval: float = 20.0,  # Very lenient - maximum seconds without blink
        required_blinks: int = 1,  # Just need 1 blink
        blink_window: float = 30.0,  # Longer time window (30 seconds)
    ) -> None:
        """
        Initialize the LBPH Face Recognizer with optional anti-spoofing and blink verification.
        
        Args:
            radius: Radius for LBP calculation (1-3, default 1)
            neighbors: Number of sample points for LBP (8, 16, or 24, default 8)
            grid_x: Number of cells in horizontal direction (4-16, default 8)
            grid_y: Number of cells in vertical direction (4-16, default 8)
            threshold: Recognition confidence threshold (lower = stricter, default 50.0)
            use_enhanced_preprocessing: Use enhanced preprocessing (alignment, CLAHE)
            enable_antispoofing: Enable anti-spoofing detection
            antispoofing_mode: Anti-spoofing mode ("basic", "high_security", "disabled")
            reject_on_spoof: Whether to reject recognition on spoofing detection
            enable_blink_verification: Enable temporal blink interval verification
            min_blink_interval: Minimum seconds between blinks (default 1.5s)
            max_blink_interval: Maximum seconds without blink (default 10.0s)
            required_blinks: Minimum blinks to verify liveness (default 2)
            blink_window: Time window to collect blinks in seconds (default 15.0s)
            
        Raises:
            ValueError: If parameters are out of valid ranges
        """
        self._validate_parameters(radius, neighbors, grid_x, grid_y, threshold)
        
        self.radius = radius
        self.neighbors = neighbors
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.threshold = threshold
        self.use_enhanced_preprocessing = use_enhanced_preprocessing
        self.enable_antispoofing = enable_antispoofing
        self.antispoofing_mode = antispoofing_mode
        self.reject_on_spoof = reject_on_spoof
        
        # Blink verification configuration
        self.enable_blink_verification = enable_blink_verification
        self.min_blink_interval = min_blink_interval
        self.max_blink_interval = max_blink_interval
        self.required_blinks = required_blinks
        self.blink_window = blink_window
        
        # Rate limiting for blink terminal output (print every 5 seconds)
        self.last_blink_print_time = 0.0
        self.blink_print_interval = 5.0  # seconds
        
        # Create LBPH recognizer
        # Note: Don't pass threshold to LBPH - we handle thresholding after prediction
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(  # type: ignore
            radius=radius,
            neighbors=neighbors,
            grid_x=grid_x,
            grid_y=grid_y
        )
        
        # Initialize preprocessor
        self.preprocessor = self._create_preprocessor() if use_enhanced_preprocessing else None
        
        # Initialize anti-spoofing detector
        self.antispoofing_detector = self._create_antispoofing_detector() if enable_antispoofing else None
        
        # Label mappings
        self.label_to_name: Dict[int, str] = {}
        self.name_to_label: Dict[str, int] = {}
        self.is_trained = False
        
        # Temporal smoothing and stabilization - PER PERSON
        self._confidence_history_per_label: Dict[int, List[float]] = {}  # Confidence history per label
        self._quality_history_per_label: Dict[int, List[float]] = {}  # Quality history per label
        self._last_stable_confidence_per_label: Dict[int, float] = {}  # Last stable confidence per label
        
        # Person stability tracking - ANTI-FLICKER
        self._current_person_label: Optional[int] = None  # Currently recognized person
        self._current_person_confidence: float = float('inf')  # Current person's best confidence
        self._consecutive_frames_per_label: Dict[int, int] = {}  # Consecutive frames per person
        
        # Blink verification tracking - PER PERSON
        self._blink_timestamps_per_label: Dict[int, List[float]] = {}  # Blink timestamps per label
        self._session_start_per_label: Dict[int, float] = {}  # Session start time per label
        self._last_blink_check_per_label: Dict[int, float] = {}  # Last check time per label
        self._blink_verified_per_label: Dict[int, bool] = {}  # Verification status per label
        self._verification_in_progress_per_label: Dict[int, bool] = {}  # Verification in progress per label

    @staticmethod
    def _validate_parameters(
        radius: int,
        neighbors: int,
        grid_x: int,
        grid_y: int,
        threshold: float
    ) -> None:
        """Validate initialization parameters."""
        if not 1 <= radius <= 3:
            raise ValueError(f"Radius must be between 1-3, got {radius}")
        if neighbors not in {8, 16, 24}:
            raise ValueError(f"Neighbors must be 8, 16, or 24, got {neighbors}")
        if not 4 <= grid_x <= 16:
            raise ValueError(f"grid_x must be between 4-16, got {grid_x}")
        if not 4 <= grid_y <= 16:
            raise ValueError(f"grid_y must be between 4-16, got {grid_y}")
        if threshold < 0:
            raise ValueError(f"Threshold must be non-negative, got {threshold}")

    def _create_preprocessor(self) -> FacePreprocessor:
        """Create and configure the face preprocessor to match training pipeline."""
        return FacePreprocessor(
            target_size=(150, 150),  # Match training pipeline
            use_clahe=True,
            clahe_clip_limit=2.0,
            clahe_tile_size=(8, 8),
            enable_alignment=True,  # Match training pipeline
            enable_landmark_alignment=True,  # Match training pipeline
            enable_photometric_norm=True,  # Match training pipeline
            enable_quality_rejection=True,  # Match training pipeline
            enable_pose_filtering=True,  # Match training pipeline
            enable_antispoofing=False,  # Disable for recognition
            quality_rejection_threshold=0.3,  # Match training pipeline
            face_crop_margin=0.2  # Match training pipeline
        )
    
    def _create_antispoofing_detector(self) -> Optional[UltimateAntiSpoof]:
        """Create and configure the anti-spoofing detector."""
        if self.antispoofing_mode == "basic":
            return UltimateAntiSpoof(level=SecurityLevel.BALANCED.value)
        elif self.antispoofing_mode == "high_security":
            return UltimateAntiSpoof(level=SecurityLevel.STRICT.value)
        else:
            return None

    def train(
        self,
        face_samples: List[np.ndarray],
        labels: List[int],
        label_names: Optional[Dict[int, str]] = None
    ) -> Dict[str, object]:
        """
        Train the face recognition model with provided samples.
        
        Args:
            face_samples: List of face images (grayscale numpy arrays)
            labels: Corresponding integer labels/IDs for each face sample
            label_names: Optional mapping of label IDs to person names
            
        Returns:
            dict: Training results containing samples processed and training info
            
        Raises:
            ValueError: If face_samples and labels have mismatched lengths or are empty
        """
        self._validate_training_data(face_samples, labels)
        
        # Preprocess all samples
        processed_samples = [self._preprocess_sample(sample) for sample in face_samples]
        
        # Store label mappings
        self._update_label_mappings(label_names)
        
        # Train the recognizer
        self.recognizer.train(processed_samples, np.array(labels))
        self.is_trained = True
        
        return {
            'samples_processed': len(processed_samples),
            'unique_labels': len(set(labels)),
            'label_mappings': self.label_to_name.copy()
        }

    def _validate_training_data(
        self,
        face_samples: List[np.ndarray],
        labels: List[int]
    ) -> None:
        """Validate training data consistency."""
        if len(face_samples) == 0:
            raise ValueError("No training samples provided")
        
        if len(face_samples) != len(labels):
            raise ValueError(
                f"Number of samples ({len(face_samples)}) and labels "
                f"({len(labels)}) must match"
            )

    def _preprocess_sample(self, sample: np.ndarray) -> np.ndarray:
        """Preprocess a single face sample with distance adaptation."""
        if self.preprocessor is not None:
            # Apply distance-adaptive preprocessing
            return self._preprocess_with_distance_adaptation(sample)
        
        # Legacy basic preprocessing
        if len(sample.shape) == 3:
            sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        return cv2.resize(sample, self.DEFAULT_FACE_SIZE)

    def _preprocess_with_distance_adaptation(self, sample: np.ndarray) -> np.ndarray:
        """
        Preprocess face sample with distance-adaptive enhancement.
        
        Adapts preprocessing based on face size to maintain consistent recognition
        regardless of distance from camera.
        """
        try:
            # Convert to grayscale if needed
            gray = self._ensure_grayscale(sample)
            
            # Calculate face size for distance assessment
            face_height, face_width = gray.shape
            face_area = face_height * face_width
            
            # Determine distance category based on face size
            if face_area < 8000:  # Small face (far away)
                return self._preprocess_small_face(gray)
            elif face_area > 20000:  # Large face (very close)
                return self._preprocess_large_face(gray)
            else:  # Medium face (optimal distance)
                if self.preprocessor:
                    return self.preprocessor.preprocess(sample, align=False, equalize=True)
                return cv2.resize(gray, self.DEFAULT_FACE_SIZE)
                
        except Exception as e:
            warnings.warn(f"Distance-adaptive preprocessing failed: {e}", RuntimeWarning)
            if self.preprocessor:
                return self.preprocessor.preprocess(sample, align=False, equalize=True)
            return cv2.resize(sample, self.DEFAULT_FACE_SIZE)

    def _preprocess_small_face(self, gray: np.ndarray) -> np.ndarray:
        """Enhanced preprocessing for small faces (far away)."""
        try:
            # Upscale before processing to improve feature detail
            scale_factor = 1.5
            upscaled = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
            
            # Apply aggressive CLAHE for small faces
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
            enhanced = clahe.apply(upscaled)
            
            # Apply unsharp masking for detail enhancement
            gaussian_blur = cv2.GaussianBlur(enhanced, (0, 0), 1.5)
            enhanced = cv2.addWeighted(enhanced, 1.5, gaussian_blur, -0.5, 0)
            
            # Resize to target size
            result = cv2.resize(enhanced, self.DEFAULT_FACE_SIZE, interpolation=cv2.INTER_AREA)
            
            # Final histogram stretching
            result = self._stretch_histogram(result)
            
            return result
            
        except Exception as e:
            warnings.warn(f"Small face preprocessing failed: {e}", RuntimeWarning)
            return cv2.resize(gray, self.DEFAULT_FACE_SIZE)

    def _preprocess_large_face(self, gray: np.ndarray) -> np.ndarray:
        """Optimized preprocessing for large faces (very close)."""
        try:
            # Downscale first to reduce noise
            scale_factor = 0.8
            downscaled = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
            
            # Apply gentle CLAHE for large faces
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(downscaled)
            
            # Light denoising
            enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
            
            # Resize to target size
            result = cv2.resize(enhanced, self.DEFAULT_FACE_SIZE, interpolation=cv2.INTER_AREA)
            
            return result
            
        except Exception as e:
            warnings.warn(f"Large face preprocessing failed: {e}", RuntimeWarning)
            return cv2.resize(gray, self.DEFAULT_FACE_SIZE)

    def _update_label_mappings(self, label_names: Optional[Dict[int, str]]) -> None:
        """Update internal label mapping dictionaries."""
        if label_names:
            self.label_to_name = label_names.copy()
            self.name_to_label = {name: label for label, name in label_names.items()}

    def update(
        self,
        face_samples: List[np.ndarray],
        labels: List[int],
        label_names: Optional[Dict[int, str]] = None
    ) -> Dict[str, object]:
        """
        Update existing model with new training samples (incremental learning).
        
        Args:
            face_samples: New face images to add
            labels: Corresponding labels for new samples
            label_names: Optional new label name mappings
            
        Returns:
            dict: Update results
            
        Raises:
            RuntimeError: If model hasn't been trained yet
            ValueError: If samples and labels length mismatch
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before updating")
        
        self._validate_training_data(face_samples, labels)
        
        # Preprocess samples
        processed_samples = [self._preprocess_sample(sample) for sample in face_samples]
        
        # Update label mappings
        if label_names:
            self.label_to_name.update(label_names)
            self.name_to_label.update(
                {name: label for label, name in label_names.items()}
            )
        
        # Update the recognizer
        self.recognizer.update(processed_samples, np.array(labels))
        
        return {
            'samples_added': len(processed_samples),
            'total_labels': len(self.label_to_name)
        }

    def predict(
        self,
        face_image: np.ndarray,
        return_confidence: bool = True
    ) -> Union[Tuple[Union[int, str], float], Union[int, str]]:
        """
        Predict the identity of a face in the given image.
        
        Args:
            face_image: Face image (grayscale or BGR)
            return_confidence: Whether to return confidence score
            
        Returns:
            If return_confidence=True: (label/name, confidence)
            If return_confidence=False: label/name only
            
        Raises:
            RuntimeError: If model hasn't been trained yet
            
        Note:
            Lower confidence = better match
            Confidence > threshold = unknown person
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        # Preprocess and predict
        processed_image = self._preprocess_sample(face_image)
        label, confidence = self.recognizer.predict(processed_image)
        
        # Convert label to name if mapping exists
        result_label = self.label_to_name.get(label, label) if label in self.label_to_name else label  # type: ignore[assignment]
        
        # Ensure confidence is always a float
        confidence_float = float(confidence) if confidence is not None else float('inf')
        
        if return_confidence:
            return (result_label, confidence_float)  # type: ignore[return-value]
        return result_label  # type: ignore[return-value]

    def predict_with_name(self, face_image: np.ndarray) -> RecognitionResult:
        """
        Ultra-strict recognition with comprehensive anti false-positive protection and anti-spoofing.
        
        This method implements the strictest possible measures to prevent
        cross-identity false positives and uncertain matches through:
        
        - Advanced lighting normalization
        - Adaptive threshold adjustment based on lighting conditions
        - Multi-stage quality assessment
        - Temporal confidence smoothing
        - Label consistency tracking
        - Anti-spoofing detection (if enabled)
        
        Args:
            face_image: Face image to recognize
            
        Returns:
            RecognitionResult object with comprehensive recognition data
            
        Raises:
            RuntimeError: If model hasn't been trained yet
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
            
        # Stage 1: Anti-spoofing detection (if enabled)
        liveness_result = None
        is_live = True
        detected_blink_for_label = None  # Track which label had a blink
        
        if self.enable_antispoofing and self.antispoofing_detector is not None:
            try:
                liveness_result = self.antispoofing_detector.check(face_image)
                is_live = liveness_result.is_real
                
                # Try to extract blink detection from UltimateAntiSpoof - IMPROVED
                if hasattr(liveness_result, 'metrics') and liveness_result.metrics is not None:
                    # Check if blink score exists and is meaningful
                    if hasattr(liveness_result.metrics, 'blink_score'):
                        blink_score = liveness_result.metrics.blink_score
                        # More lenient threshold - consider any blink indication
                        if blink_score > 0.3:  # Lowered from 0.7 for better detection
                            import time
                            # Don't record blink yet - we need to know which person it is first
                            detected_blink_for_label = time.time()
                            # Rate-limited output
                            if self.enable_blink_verification and (detected_blink_for_label - self.last_blink_print_time >= self.blink_print_interval):
                                self.last_blink_print_time = detected_blink_for_label
                                print(f"  [Blink detected] Score: {blink_score:.2f} | Pending user assignment...")
                
                if not is_live and self.reject_on_spoof:
                    return self._create_spoof_result(liveness_result)
            except Exception as e:
                print(f"Warning: Anti-spoofing check failed: {e}")
        
        # Stage 2: Lighting normalization
        normalized_face = self._normalize_lighting(face_image)
        
        # Stage 3: Assess lighting and quality
        lighting_metrics = self._assess_lighting_condition(face_image)
        quality_score = self._assess_face_quality(normalized_face)
        
        # Stage 4: Early rejection for very low quality
        if quality_score < self.MIN_QUALITY_THRESHOLD:
            return self._create_unknown_result(quality_score, lighting_metrics, liveness_result, is_live)
        
        # Stage 5: Perform recognition
        prediction_result = self.predict(normalized_face, return_confidence=True)
        
        # Ensure we have valid types from prediction
        if isinstance(prediction_result, tuple):
            # predict returned (label, confidence)
            label, confidence = prediction_result
            confidence = float(confidence) if confidence is not None else float('inf')
        else:
            # predict returned just label
            label = prediction_result
            confidence = float('inf')
        
        label_id, name = self._resolve_label(label)
        
        # NOW that we know the label, record the blink for THIS specific person
        import time
        current_time = time.time()
        if detected_blink_for_label is not None:
            self._record_blink(label_id, detected_blink_for_label)
            # Get blink count for this user
            user_blink_count = len(self._blink_timestamps_per_label.get(label_id, []))
            # Rate-limited output
            if self.enable_blink_verification and (current_time - self.last_blink_print_time >= self.blink_print_interval):
                self.last_blink_print_time = current_time
                print(f"  [✓ Blink] User: {name} | Total blinks for {name}: {user_blink_count} | Time: {detected_blink_for_label:.2f}")
        
        # Stage 5.5: Verify blink timing FOR THIS SPECIFIC PERSON (advisory layer)
        blink_warning = None
        
        if self.enable_blink_verification:
            blink_valid, blink_reason = self._verify_blink_timing(label_id, current_time)
            
            if not blink_valid:
                # Log warning but don't reject - blink detection is experimental
                blink_warning = f"Blink verification warning for {name}: {blink_reason}"
                # Rate-limited output
                if current_time - self.last_blink_print_time >= self.blink_print_interval:
                    self.last_blink_print_time = current_time
                    print(f"  ⚠️  {blink_warning}")
                # Continue with recognition instead of rejecting
        
        # Stage 6: Apply confidence smoothing with distance adaptation
        face_size = (normalized_face.shape[1], normalized_face.shape[0])  # (width, height)
        # Ensure confidence is float before smoothing
        confidence_float = float(confidence) if confidence is not None else float('inf')
        smoothed_confidence = self._smooth_confidence(confidence_float, quality_score, label_id, face_size)
        
        # Stage 7: Adjust confidence based on liveness result
        if liveness_result and not is_live:
            # Reduce confidence for spoofed faces
            smoothed_confidence *= (1.0 - liveness_result.confidence)
        elif liveness_result and not liveness_result.is_real:
            # Slightly reduce confidence for suspicious faces
            smoothed_confidence *= 0.9
        
        # Stage 8: Calculate adaptive threshold
        adaptive_threshold = self._calculate_adaptive_threshold(lighting_metrics)
        
        # Stage 9: Multi-stage recognition decision
        is_recognized = self._evaluate_recognition(
            smoothed_confidence,
            adaptive_threshold,
            quality_score,
            label_id
        )
        
        return RecognitionResult(
            recognized=is_recognized,
            label=label_id,
            name=name,
            confidence=smoothed_confidence,
            raw_confidence=confidence,
            threshold=adaptive_threshold,
            quality_score=quality_score,
            lighting_metrics=lighting_metrics,
            liveness_result=liveness_result,
            is_live=is_live
        )

    def _create_unknown_result(
        self,
        quality_score: float,
        lighting_metrics: LightingMetrics,
        liveness_result: Optional[object] = None,
        is_live: bool = True
    ) -> RecognitionResult:
        """Create a result object for unknown/rejected faces."""
        return RecognitionResult(
            recognized=False,
            label=-1,
            name='Unknown',
            confidence=float('inf'),
            raw_confidence=float('inf'),
            threshold=self.threshold,
            quality_score=quality_score,
            lighting_metrics=lighting_metrics,
            liveness_result=liveness_result,  # type: ignore
            is_live=is_live
        )
    
    def _create_spoof_result(self, liveness_result: AntiSpoofResult) -> RecognitionResult:
        """Create a result object for spoofed faces."""
        return RecognitionResult(
            recognized=False,
            label=-1,
            name=f'Spoof: {liveness_result.attack_type.value}',
            confidence=float('inf'),
            raw_confidence=float('inf'),
            threshold=self.threshold,
            quality_score=0.0,
            lighting_metrics=LightingMetrics(
                condition=LightingCondition.UNKNOWN,
                brightness=127.0,
                contrast=50.0,
                quality=0.0
            ),
            liveness_result=liveness_result,
            is_live=False
        )

    def _resolve_label(self, label: Union[int, str]) -> Tuple[int, str]:
        """Resolve label to both ID and name."""
        if isinstance(label, int):
            name = self.label_to_name.get(label, f"Unknown ({label})")
            label_id = label
        else:
            name = label
            label_id = self.name_to_label.get(label, -1)
        return label_id, name

    def _normalize_lighting(self, face_image: np.ndarray) -> np.ndarray:
        """
        Advanced lighting normalization to reduce lighting sensitivity.
        
        Applies multiple techniques:
        1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        2. Gamma correction for brightness normalization
        3. Histogram stretching to full dynamic range
        4. Gaussian blur for noise reduction
        
        Args:
            face_image: Face image to normalize
            
        Returns:
            Lighting-normalized face image
        """
        try:
            # Convert to grayscale if needed
            gray = self._ensure_grayscale(face_image)
            
            # Apply CLAHE for local contrast enhancement - ROBUST
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6, 6))  # Balanced for robustness
            normalized = clahe.apply(gray)
            
            # Gamma correction for brightness normalization - ROBUST
            gamma = 1.3  # More conservative gamma correction for robustness
            normalized = np.power(normalized / 255.0, gamma) * 255.0
            normalized = np.uint8(normalized)
            
            # Histogram stretching to full range
            # Cast to array to avoid scalar type issues
            normalized_array = np.asarray(normalized, dtype=np.uint8)
            normalized = self._stretch_histogram(normalized_array)
            
            # Gaussian blur to reduce noise while preserving features
            normalized = cv2.GaussianBlur(normalized, (3, 3), 0)
            
            return normalized
            
        except Exception as e:
            warnings.warn(f"Lighting normalization failed: {e}", RuntimeWarning)
            return face_image

    @staticmethod
    def _ensure_grayscale(image: np.ndarray) -> np.ndarray:
        """Ensure image is grayscale."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.copy()

    @staticmethod
    def _stretch_histogram(image: np.ndarray) -> np.ndarray:
        """Stretch histogram to full 0-255 range."""
        # Ensure image is numpy array, not uint8 scalar
        if not isinstance(image, np.ndarray):
            return np.array(image, dtype=np.uint8)
        
        # Convert to float for calculations
        img_float = image.astype(float)
        min_val = np.min(img_float)
        max_val = np.max(img_float)
        
        if max_val > min_val:
            stretched = ((img_float - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            return stretched
        return image

    def _assess_lighting_condition(self, face_image: np.ndarray) -> LightingMetrics:
        """
        Assess the lighting condition of a face image.
        
        Args:
            face_image: Face image to assess
            
        Returns:
            LightingMetrics object with lighting information
        """
        try:
            gray = self._ensure_grayscale(face_image)
            
            # Calculate lighting metrics
            brightness = float(np.mean(gray))
            contrast = float(np.std(gray))
            
            # Determine lighting condition
            if brightness < 80:
                condition = LightingCondition.DIM
            elif brightness > 180:
                condition = LightingCondition.BRIGHT
            else:
                condition = LightingCondition.NORMAL
            
            # Calculate lighting quality score
            brightness_score = 1.0 - abs(brightness - 127) / 127
            contrast_score = min(contrast / 50, 1.0)
            lighting_quality = (brightness_score + contrast_score) / 2
            
            return LightingMetrics(
                condition=condition,
                brightness=brightness,
                contrast=contrast,
                quality=lighting_quality
            )
            
        except Exception as e:
            warnings.warn(f"Lighting assessment failed: {e}", RuntimeWarning)
            return LightingMetrics(
                condition=LightingCondition.UNKNOWN,
                brightness=127.0,
                contrast=50.0,
                quality=0.5
            )

    def _assess_face_quality(self, face_image: np.ndarray) -> float:
        """
        Enhanced face quality assessment based on research recommendations.
        
        Returns quality score 0.0-1.0 (higher = better quality).
        
        Research-based criteria:
        - Reference photos: quality score 80+ recommended
        - Input images: quality score 40+ recommended
        
        Factors assessed:
        1. Brightness (20% weight)
        2. Contrast (20% weight)
        3. Edge density/sharpness (20% weight)
        4. Face symmetry (15% weight)
        5. Blur detection (15% weight)
        6. Face size (10% weight)
        
        Args:
            face_image: Face image to assess
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        try:
            gray = self._ensure_grayscale(face_image)
            gray = cv2.resize(gray, self.DEFAULT_FACE_SIZE)
            
            quality_score = 0.0
            
            # 1. Brightness check (20% weight)
            quality_score += self._assess_brightness(gray)
            
            # 2. Contrast check (20% weight)
            quality_score += self._assess_contrast(gray)
            
            # 3. Edge density (20% weight)
            quality_score += self._assess_sharpness(gray)
            
            # 4. Face symmetry (15% weight)
            quality_score += self._assess_symmetry(gray)
            
            # 5. Blur detection (15% weight)
            quality_score += self._assess_blur(gray)
            
            # 6. Face size assessment (10% weight)
            quality_score += self._assess_face_size(gray)
            
            return min(quality_score, 1.0)
            
        except Exception as e:
            warnings.warn(f"Quality assessment failed: {e}", RuntimeWarning)
            return 0.0

    @staticmethod
    def _assess_brightness(gray: np.ndarray) -> float:
        """Assess brightness quality (0.0-0.2) - ROBUST."""
        mean_brightness = np.mean(gray)
        if 30 <= mean_brightness <= 200:  # More lenient optimal range for robustness
            return 0.2
        elif 20 <= mean_brightness <= 220:  # More lenient good range
            return 0.15
        elif 15 <= mean_brightness <= 240:  # More lenient acceptable range
            return 0.1
        return 0.0

    @staticmethod
    def _assess_contrast(gray: np.ndarray) -> float:
        """Assess contrast quality (0.0-0.2) - ROBUST."""
        contrast = np.std(gray)
        if contrast > 30:  # More lenient contrast requirement for robustness
            return 0.2
        elif contrast > 25:  # More lenient contrast requirement
            return 0.15
        elif contrast > 20:  # More lenient contrast requirement
            return 0.1
        return 0.0

    @staticmethod
    def _assess_sharpness(gray: np.ndarray) -> float:
        """Assess sharpness via edge density (0.0-0.2) - ROBUST."""
        edges = cv2.Canny(gray, 50, 150)  # Balanced edge detection for robustness
        edge_density = np.sum(edges > 0) / edges.size
        
        if edge_density > 0.12:  # More lenient edge density requirement for robustness
            return 0.2
        elif edge_density > 0.08:  # More lenient edge density requirement
            return 0.15
        elif edge_density > 0.05:  # More lenient edge density requirement
            return 0.1
        return 0.0

    @staticmethod
    def _assess_symmetry(gray: np.ndarray) -> float:
        """Assess face symmetry (0.0-0.15)."""
        mid = gray.shape[1] // 2
        left_half = gray[:, :mid]
        right_half = cv2.flip(gray[:, mid:], 1)
        
        # Ensure same size
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]
        
        symmetry_diff = np.mean(np.abs(
            left_half.astype(float) - right_half.astype(float)
        ))
        
        if symmetry_diff < 15:
            return 0.15
        elif symmetry_diff < 25:
            return 0.1
        elif symmetry_diff < 35:
            return 0.05
        return 0.0

    @staticmethod
    def _assess_blur(gray: np.ndarray) -> float:
        """Assess blur using Laplacian variance (0.0-0.15)."""
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if blur_score > 150:
            return 0.15
        elif blur_score > 100:
            return 0.1
        elif blur_score > 50:
            return 0.05
        return 0.0

    @staticmethod
    def _assess_face_size(gray: np.ndarray) -> float:
        """Assess face size (0.0-0.1)."""
        face_area = gray.shape[0] * gray.shape[1]
        
        if face_area >= 8000:
            return 0.1
        elif face_area >= 5000:
            return 0.05
        else:
            return 0.0

    def _smooth_confidence(
        self,
        confidence: float,
        quality_score: float,
        label: int,
        face_size: Optional[Tuple[int, int]] = None
    ) -> float:
        """
        Advanced confidence smoothing with temporal consistency and distance adaptation - PER PERSON.
        
        Implements:
        - Weighted moving average (recent samples weighted more)
        - Median filtering for outlier rejection
        - Temporal stability enforcement
        - Quality-based adjustment
        - Distance-adaptive confidence adjustment
        - SEPARATE tracking for each person (label_id)
        
        Args:
            confidence: Raw confidence score
            quality_score: Face quality score (0.0-1.0)
            label: Predicted label ID
            face_size: Optional face size (width, height) for distance adaptation
            
        Returns:
            Stabilized confidence score
        """
        # Initialize per-person tracking if needed
        if label not in self._confidence_history_per_label:
            self._confidence_history_per_label[label] = []
            self._quality_history_per_label[label] = []
            # Don't initialize last_stable_confidence yet - will be set after first smoothing
        
        # Update histories FOR THIS PERSON ONLY
        self._confidence_history_per_label[label].append(confidence)
        self._quality_history_per_label[label].append(quality_score)
        
        # Maintain history size limits FOR THIS PERSON
        self._trim_history_for_label(label)
        
        # Get this person's history
        conf_history = self._confidence_history_per_label[label]
        qual_history = self._quality_history_per_label[label]
        
        # Need at least 3 samples for proper smoothing
        if len(conf_history) < 3:
            return confidence
        
        # Calculate weighted moving average FOR THIS PERSON
        weights = np.linspace(0.3, 1.0, len(conf_history))
        weighted_conf = float(np.average(conf_history, weights=weights))
        
        # Calculate median for outlier rejection FOR THIS PERSON
        median_conf = float(np.median(conf_history))
        
        # Apply temporal stability FOR THIS PERSON
        smoothed_conf = self._apply_temporal_stability_for_label(weighted_conf, label)
        
        # Reject outliers using median
        smoothed_conf = self._reject_outliers(smoothed_conf, median_conf)
        
        # Quality-based adjustment - OPTIMIZED for better confidence
        smoothed_conf = self._apply_quality_adjustment_for_label(smoothed_conf, label)
        
        # Distance-adaptive confidence adjustment
        if face_size is not None:
            smoothed_conf = self._apply_distance_adjustment(smoothed_conf, face_size)
        
        # Additional confidence reduction for high quality images - CONSERVATIVE
        if quality_score > self.HIGH_QUALITY_THRESHOLD:
            smoothed_conf *= 0.9  # Less aggressive reduction to prevent false positives
        
        # Label consistency bonus - CONSERVATIVE (check last 3 of THIS PERSON)
        if len(conf_history) >= 3:
            smoothed_conf = (
                smoothed_conf * 0.8 + 
                np.mean(conf_history[-3:]) * 0.2
            )
            # Less aggressive additional reduction for consistent recognition
            smoothed_conf *= 0.95
        
        # Final bounds checking
        smoothed_conf = max(0.0, min(smoothed_conf, 200.0))
        
        # Update stable confidence FOR THIS PERSON
        self._last_stable_confidence_per_label[label] = float(smoothed_conf)
        
        return float(smoothed_conf)
    
    def _reject_outliers(self, smoothed_conf: float, median_conf: float) -> float:
        """Reject outlier confidence values using median."""
        if abs(smoothed_conf - median_conf) > 20:
            return smoothed_conf * 0.4 + median_conf * 0.6
        return smoothed_conf

    def _trim_history_for_label(self, label: int) -> None:
        """Trim history lists to maximum size for a specific person."""
        if len(self._confidence_history_per_label[label]) > self.MAX_CONFIDENCE_HISTORY:
            self._confidence_history_per_label[label].pop(0)
        if len(self._quality_history_per_label[label]) > self.MAX_QUALITY_HISTORY:
            self._quality_history_per_label[label].pop(0)

    def _apply_temporal_stability_for_label(self, weighted_conf: float, label: int) -> float:
        """Apply temporal stability constraints for a specific person."""
        last_stable = self._last_stable_confidence_per_label.get(label, None)
        if last_stable is None:
            return weighted_conf
        
        change_magnitude = abs(weighted_conf - last_stable)
        
        if change_magnitude > self.CONFIDENCE_STABILITY_THRESHOLD:
            # Large change detected - use conservative smoothing
            return weighted_conf * 0.3 + last_stable * 0.7
        
        return weighted_conf

    def _apply_quality_adjustment_for_label(self, smoothed_conf: float, label: int) -> float:
        """Adjust confidence based on average quality for a specific person - CONSERVATIVE."""
        qual_history = self._quality_history_per_label.get(label, [])
        if not qual_history:
            return smoothed_conf
        
        avg_quality = np.mean(qual_history)
        
        if avg_quality > self.HIGH_QUALITY_THRESHOLD:
            return smoothed_conf * 0.9  # Less aggressive reduction to prevent false positives
        elif avg_quality < self.LOW_QUALITY_THRESHOLD:
            return smoothed_conf * 1.1  # Less aggressive increase for low quality
        
        return smoothed_conf

    def _apply_distance_adjustment(self, smoothed_conf: float, face_size: Tuple[int, int]) -> float:
        """
        WEBCAM-OPTIMIZED distance adjustment for 720p.
        
        Optimized for typical webcam face sizes (720p: 3000-40000 pixels):
        - Very far (<6000): 25% improvement
        - Far (<12000): 15% improvement  
        - Medium-far (<18000): 8% improvement
        - Optimal (<25000): No adjustment (baseline)
        - Close (<35000): 12% improvement
        - Very close (>35000): 20% improvement
        """
        try:
            face_width, face_height = face_size
            face_area = face_width * face_height
            
            # WEBCAM-SPECIFIC thresholds (720p faces: 3000-40000 pixels)
            if face_area < 6000:  # Very far
                return smoothed_conf * 0.75  # 25% improvement
            elif face_area < 12000:  # Far
                return smoothed_conf * 0.85  # 15% improvement
            elif face_area < 18000:  # Medium-far
                return smoothed_conf * 0.92  # 8% improvement
            elif face_area < 25000:  # Optimal - no adjustment
                return smoothed_conf
            elif face_area < 35000:  # Close
                return smoothed_conf * 0.88  # 12% improvement
            else:  # Very close
                return smoothed_conf * 0.80  # 20% improvement
                
        except Exception as e:
            warnings.warn(f"Distance adjustment failed: {e}", RuntimeWarning)
            return smoothed_conf

    def _calculate_adaptive_threshold(
        self,
        lighting_metrics: LightingMetrics
    ) -> float:
        """
        Calculate adaptive threshold based on lighting conditions.
        
        Args:
            lighting_metrics: Lighting assessment metrics
            
        Returns:
            Adjusted threshold value
        """
        multiplier = {
            LightingCondition.DIM: 1.2,      # More lenient for dim lighting
            LightingCondition.BRIGHT: 0.9,   # Stricter for bright lighting
            LightingCondition.NORMAL: 1.0,   # Standard threshold
            LightingCondition.UNKNOWN: 1.0   # Standard threshold
        }
        
        return self.threshold * multiplier[lighting_metrics.condition]

    def _evaluate_recognition(
        self,
        confidence: float,
        threshold: float,
        quality_score: float,
        label_id: int
    ) -> bool:
        """
        Multi-stage recognition evaluation with ULTRA-STRICT anti-false-positive and anti-flicker protection.
        
        Args:
            confidence: Smoothed confidence score (LBPH: lower = better match)
            threshold: Adaptive threshold
            quality_score: Face quality score
            label_id: Predicted label ID
            
        Returns:
            True if face should be recognized, False otherwise
        """
        # Stage 1: Confidence cutoff - reject if confidence > 50 (balanced for unknowns)
        # In LBPH, lower confidence means better match
        # Confidence > 50 means poor match, likely different person or unknown
        # This prevents false positives like recognizing random people as known users
        if confidence > 50:
            return False
        
        # Stage 2: Primary threshold check
        if confidence >= threshold:
            return False
        
        # Stage 3: Ultra-strict confidence bound
        if confidence > 100:
            return False
        
        # Stage 4: Reject uncertain range
        if 90 <= confidence <= 110:
            return False
        
        # Stage 5: Quality-based rejection for high confidence
        if confidence > 45 and quality_score < 0.50:  # Quality requirement for borderline cases
            return False
        
        # Stage 6: Minimum quality requirement
        if quality_score < 0.30:  # More lenient quality threshold
            return False
        
        # Stage 7: Anti-flicker - require minimum samples before recognizing NEW person
        conf_history = self._confidence_history_per_label.get(label_id, [])
        if len(conf_history) < self.MIN_SAMPLES_FOR_RECOGNITION:
            # Not enough samples for this person yet
            return False
        
        # Stage 8: Anti-flicker - if we have a current person, require significant improvement to switch
        if self._current_person_label is not None and self._current_person_label != label_id:
            # Trying to switch to a different person
            # New person must be SIGNIFICANTLY better (lower confidence = better)
            if confidence > (self._current_person_confidence - self.PERSON_SWITCH_THRESHOLD):
                # New person is not significantly better - stay with current person
                return False
        
        # Stage 9: Update consecutive frames counter
        if label_id not in self._consecutive_frames_per_label:
            self._consecutive_frames_per_label[label_id] = 0
        self._consecutive_frames_per_label[label_id] += 1
        
        # Reset other people's consecutive frames
        for other_label in list(self._consecutive_frames_per_label.keys()):
            if other_label != label_id:
                self._consecutive_frames_per_label[other_label] = 0
        
        # Stage 10: Require consecutive frames for NEW person (not current)
        if self._current_person_label is not None and self._current_person_label != label_id:
            # Switching to new person - require 3 consecutive frames
            if self._consecutive_frames_per_label[label_id] < 3:
                return False
        
        # All checks passed - update current person tracking
        self._current_person_label = label_id
        self._current_person_confidence = min(confidence, self._current_person_confidence) if label_id == self._current_person_label else confidence
        
        return True
    
    def _verify_blink_timing(self, label_id: int, current_time: float) -> Tuple[bool, str]:
        """
        Verify that blinks occur within acceptable time intervals FOR A SPECIFIC PERSON.
        
        LENIENT MODE: This is advisory only - warns but doesn't reject.
        Blink detection is experimental and may have false positives.
        
        Args:
            label_id: The label ID of the person being verified
            current_time: Current timestamp in seconds
            
        Returns:
            Tuple of (is_valid, reason) where is_valid indicates if blink timing is acceptable
        """
        import time
        
        # Initialize per-person tracking if needed
        if label_id not in self._session_start_per_label:
            self._session_start_per_label[label_id] = current_time
            self._last_blink_check_per_label[label_id] = current_time
            self._blink_verified_per_label[label_id] = False
            self._verification_in_progress_per_label[label_id] = True
            self._blink_timestamps_per_label[label_id] = []
        
        session_duration = current_time - self._session_start_per_label[label_id]
        blink_timestamps = self._blink_timestamps_per_label.get(label_id, [])
        
        # If we're using the UltimateAntiSpoof detector, it handles blinks internally
        if self.antispoofing_detector is not None:
            # Trust the main anti-spoofing system
            return True, "Blink verification delegated to UltimateAntiSpoof"
        
        # LENIENT: If no blinks detected yet and still early, just return OK
        if len(blink_timestamps) == 0 and session_duration < self.blink_window:
            return True, f"Waiting for blinks for this person... ({session_duration:.1f}s / {self.blink_window:.1f}s)"
        
        # Clean up old blink timestamps outside the verification window
        self._blink_timestamps_per_label[label_id] = [
            ts for ts in blink_timestamps 
            if current_time - ts <= self.blink_window
        ]
        blink_timestamps = self._blink_timestamps_per_label[label_id]
        
        # Check if we're still in the verification window
        if session_duration < self.blink_window:
            # Still collecting blinks
            if len(blink_timestamps) >= self.required_blinks:
                # LENIENT: Only warn about very suspicious patterns, don't reject minor issues
                for i in range(1, len(blink_timestamps)):
                    interval = blink_timestamps[i] - blink_timestamps[i-1]
                    
                    # Only flag extremely suspicious patterns (very strict thresholds)
                    if interval < 0.3:  # Impossibly fast (< 300ms)
                        return False, f"Blinks impossibly fast ({interval:.1f}s) - likely video replay"
                    
                    # Don't check max interval - too unreliable
                
                self._blink_verified_per_label[label_id] = True
                return True, f"✓ Blink timing OK for this person ({len(blink_timestamps)} blinks detected)"
            else:
                # Still collecting, not enough blinks yet - this is OK
                return True, f"Collecting blinks for this person ({len(blink_timestamps)}/{self.required_blinks}) - {session_duration:.1f}s elapsed"
        else:
            # Verification window expired - LENIENT: Don't fail, just note it
            if not self._blink_verified_per_label.get(label_id, False):
                if len(blink_timestamps) < self.required_blinks:
                    # LENIENT: Just warn, don't reject
                    return True, f"⚠️ Only {len(blink_timestamps)}/{self.required_blinks} blinks for this person in {self.blink_window}s (advisory only)"
            
            return True, "✓ Verification complete" if self._blink_verified_per_label.get(label_id, False) else "⚠️ Insufficient data (advisory only)"
    
    def _record_blink(self, label_id: int, current_time: float) -> None:
        """
        Record a detected blink with timestamp FOR A SPECIFIC PERSON.
        
        Args:
            label_id: The label ID of the person who blinked
            current_time: Timestamp when blink was detected
        """
        # Initialize tracking for this label if needed
        if label_id not in self._blink_timestamps_per_label:
            self._blink_timestamps_per_label[label_id] = []
        
        self._blink_timestamps_per_label[label_id].append(current_time)
        
        # Keep only recent blinks within the verification window
        self._blink_timestamps_per_label[label_id] = [
            ts for ts in self._blink_timestamps_per_label[label_id] 
            if current_time - ts <= self.blink_window
        ]
    
    def reset_blink_verification(self) -> None:
        """Reset blink verification state for ALL people (start new session)."""
        self._blink_timestamps_per_label.clear()
        self._session_start_per_label.clear()
        self._last_blink_check_per_label.clear()
        self._blink_verified_per_label.clear()
        self._verification_in_progress_per_label.clear()
    
    def reset_blink_verification_for_person(self, label_id: int) -> None:
        """Reset blink verification state for a specific person."""
        if label_id in self._blink_timestamps_per_label:
            del self._blink_timestamps_per_label[label_id]
        if label_id in self._session_start_per_label:
            del self._session_start_per_label[label_id]
        if label_id in self._last_blink_check_per_label:
            del self._last_blink_check_per_label[label_id]
        if label_id in self._blink_verified_per_label:
            del self._blink_verified_per_label[label_id]
        if label_id in self._verification_in_progress_per_label:
            del self._verification_in_progress_per_label[label_id]

    def reset_confidence_history(self) -> None:
        """Reset confidence smoothing history and blink verification for a fresh start."""
        self._confidence_history_per_label.clear()
        self._quality_history_per_label.clear()
        self._last_stable_confidence_per_label.clear()
        self._current_person_label = None
        self._current_person_confidence = float('inf')
        self._consecutive_frames_per_label.clear()
        self.reset_blink_verification()

    def save_model(
        self,
        model_path: Union[str, Path],
        metadata_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, str]:
        """
        Save the trained model to file.
        
        Args:
            model_path: Path to save the LBPH model (.yml or .xml)
            metadata_path: Optional path to save metadata (label mappings)
                          If None, uses model_path with .pkl extension
            
        Returns:
            dict with saved file paths
            
        Raises:
            RuntimeError: If model hasn't been trained
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")
        
        # Convert to Path objects
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save LBPH model
        self.recognizer.save(str(model_path))
        
        # Determine metadata path
        if metadata_path is None:
            metadata_path = model_path.with_suffix('.pkl')
        else:
            metadata_path = Path(metadata_path)
        
        # Save metadata
        metadata = {
            'label_to_name': self.label_to_name,
            'name_to_label': self.name_to_label,
            'radius': self.radius,
            'neighbors': self.neighbors,
            'grid_x': self.grid_x,
            'grid_y': self.grid_y,
            'threshold': self.threshold,
            'use_enhanced_preprocessing': self.use_enhanced_preprocessing,
            'enable_antispoofing': self.enable_antispoofing,
            'antispoofing_mode': self.antispoofing_mode,
            'reject_on_spoof': self.reject_on_spoof,
            'version': '2.1'
        }
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        return {
            'model_path': str(model_path),
            'metadata_path': str(metadata_path)
        }

    def load_model(
        self,
        model_path: Union[str, Path],
        metadata_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, object]:
        """
        Load a pre-trained model from file.
        
        Args:
            model_path: Path to the saved LBPH model
            metadata_path: Optional path to metadata file
                          If None, looks for .pkl file with same name as model
            
        Returns:
            dict with loaded model information
            
        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load LBPH model
        self.recognizer.read(str(model_path))
        self.is_trained = True
        
        # Determine metadata path
        if metadata_path is None:
            metadata_path = model_path.with_suffix('.pkl')
        else:
            metadata_path = Path(metadata_path)
        
        # Load metadata if exists
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            # Restore configuration
            self.label_to_name = metadata.get('label_to_name', {})
            self.name_to_label = metadata.get('name_to_label', {})
            self.radius = metadata.get('radius', self.radius)
            self.neighbors = metadata.get('neighbors', self.neighbors)
            self.grid_x = metadata.get('grid_x', self.grid_x)
            self.grid_y = metadata.get('grid_y', self.grid_y)
            self.threshold = metadata.get('threshold', self.threshold)
            self.use_enhanced_preprocessing = metadata.get(
                'use_enhanced_preprocessing', True
            )
            self.enable_antispoofing = metadata.get('enable_antispoofing', False)
            self.antispoofing_mode = metadata.get('antispoofing_mode', 'basic')
            self.reject_on_spoof = metadata.get('reject_on_spoof', True)
            
            # Reinitialize preprocessor if needed
            if self.use_enhanced_preprocessing and self.preprocessor is None:
                self.preprocessor = self._create_preprocessor()
            
            # Reinitialize anti-spoofing detector if needed
            if self.enable_antispoofing and self.antispoofing_detector is None:
                self.antispoofing_detector = self._create_antispoofing_detector()
            
            return {
                'model_loaded': str(model_path),
                'metadata_loaded': str(metadata_path),
                'total_labels': len(self.label_to_name),
                'label_mappings': self.label_to_name.copy(),
                'version': metadata.get('version', '1.0')
            }
        else:
            warnings.warn(
                f"Metadata file not found: {metadata_path}. "
                "Label mappings not available.",
                RuntimeWarning
            )
            return {
                'model_loaded': str(model_path),
                'metadata_loaded': None,
                'warning': 'No metadata file found, label mappings not available'
            }

    def get_known_people(self) -> List[str]:
        """
        Get list of known people (names) in the trained model.
        
        Returns:
            List of person names sorted alphabetically
        """
        return sorted(self.name_to_label.keys())

    def get_label_mapping(self) -> Dict[int, str]:
        """
        Get the label-to-name mapping.
        
        Returns:
            Dictionary mapping label IDs to person names
        """
        return self.label_to_name.copy()

    def get_model_info(self) -> Dict[str, object]:
        """
        Get comprehensive model information and statistics.
        
        Returns:
            Dictionary containing model configuration and state
        """
        return {
            'is_trained': self.is_trained,
            'total_people': len(self.label_to_name),
            'known_people': self.get_known_people(),
            'parameters': {
                'radius': self.radius,
                'neighbors': self.neighbors,
                'grid_x': self.grid_x,
                'grid_y': self.grid_y,
                'threshold': self.threshold,
                'use_enhanced_preprocessing': self.use_enhanced_preprocessing,
                'enable_antispoofing': self.enable_antispoofing,
                'antispoofing_mode': self.antispoofing_mode,
                'reject_on_spoof': self.reject_on_spoof
            },
            'history_sizes': {
                'tracked_people': len(self._confidence_history_per_label),
                'total_confidence_samples': sum(len(h) for h in self._confidence_history_per_label.values()),
                'total_quality_samples': sum(len(h) for h in self._quality_history_per_label.values())
            }
        }

    def set_threshold(self, threshold: float) -> None:
        """
        Update the recognition threshold.
        
        Args:
            threshold: New threshold value (lower = stricter)
            
        Raises:
            ValueError: If threshold is negative
        """
        if threshold < 0:
            raise ValueError(f"Threshold must be non-negative, got {threshold}")
        self.threshold = threshold

    def get_threshold(self) -> float:
        """
        Get the current recognition threshold.
        
        Returns:
            Current threshold value
        """
        return self.threshold
    
    def enable_antispoofing_detection(self, mode: str = "basic", reject_on_spoof: bool = True) -> None:
        """
        Enable anti-spoofing detection.
        
        Args:
            mode: Anti-spoofing mode ("basic", "high_security")
            reject_on_spoof: Whether to reject recognition on spoofing detection
        """
        self.enable_antispoofing = True
        self.antispoofing_mode = mode
        self.reject_on_spoof = reject_on_spoof
        self.antispoofing_detector = self._create_antispoofing_detector()
    
    def disable_antispoofing_detection(self) -> None:
        """Disable anti-spoofing detection."""
        self.enable_antispoofing = False
        self.antispoofing_detector = None
    
    def enable_blink_verification_detection(
        self,
        min_blink_interval: float = 0.5,  # Very lenient
        max_blink_interval: float = 20.0,  # Very lenient
        required_blinks: int = 1,  # Just need 1 blink
        blink_window: float = 30.0  # Longer window
    ) -> None:
        """
        Enable temporal blink verification for additional liveness detection.
        
        WARNING: This is an EXPERIMENTAL feature that may have false positives.
        It runs in advisory mode - warns but doesn't reject recognition.
        
        Args:
            min_blink_interval: Minimum seconds between blinks (default 0.5s - very lenient)
            max_blink_interval: Maximum seconds without blink (default 20.0s - very lenient)
            required_blinks: Minimum blinks to verify liveness (default 1 - very lenient)
            blink_window: Time window to collect blinks in seconds (default 30.0s - longer)
        """
        self.enable_blink_verification = True
        self.min_blink_interval = min_blink_interval
        self.max_blink_interval = max_blink_interval
        self.required_blinks = required_blinks
        self.blink_window = blink_window
        self.reset_blink_verification()
        print(f"✓ Blink verification enabled (ADVISORY MODE)")
        print(f"  Parameters: {required_blinks} blinks in {blink_window}s window")
    
    def disable_blink_verification_detection(self) -> None:
        """Disable temporal blink verification."""
        self.enable_blink_verification = False
        self.reset_blink_verification()
    
    def get_blink_verification_info(self, label_id: Optional[int] = None) -> Dict:
        """
        Get blink verification configuration and status.
        
        Args:
            label_id: Optional label ID to get info for specific person.
                     If None, returns overall system info.
        
        Returns:
            Dictionary with blink verification settings and current status
        """
        base_info = {
            'enabled': self.enable_blink_verification,
            'min_blink_interval': self.min_blink_interval,
            'max_blink_interval': self.max_blink_interval,
            'required_blinks': self.required_blinks,
            'blink_window': self.blink_window,
            'total_tracked_people': len(self._blink_timestamps_per_label)
        }
        
        if label_id is not None and label_id in self._blink_timestamps_per_label:
            # Return info for specific person
            base_info.update({
                'label_id': label_id,
                'current_blinks': len(self._blink_timestamps_per_label[label_id]),
                'blink_verified': self._blink_verified_per_label.get(label_id, False),
                'verification_in_progress': self._verification_in_progress_per_label.get(label_id, False),
                'session_duration': time.time() - self._session_start_per_label[label_id] if label_id in self._session_start_per_label else 0.0
            })
        elif label_id is not None:
            # Person not tracked yet
            base_info.update({
                'label_id': label_id,
                'current_blinks': 0,
                'blink_verified': False,
                'verification_in_progress': False,
                'session_duration': 0.0,
                'note': 'Person not yet tracked'
            })
        
        return base_info
    
    def get_all_user_blink_stats(self) -> Dict[str, Dict]:
        """
        Get blink statistics for all tracked users.
        
        Returns:
            Dictionary mapping user names to their blink statistics
        """
        stats = {}
        for label_id, timestamps in self._blink_timestamps_per_label.items():
            # Get name from label_to_name dict
            name = self.label_to_name.get(label_id, f"Unknown_{label_id}")
            stats[name] = {
                'label_id': label_id,
                'total_blinks': len(timestamps),
                'blink_verified': self._blink_verified_per_label.get(label_id, False),
                'verification_in_progress': self._verification_in_progress_per_label.get(label_id, False),
                'session_duration': time.time() - self._session_start_per_label[label_id] if label_id in self._session_start_per_label else 0.0,
                'recent_blink_timestamps': timestamps[-5:] if len(timestamps) > 0 else []  # Last 5 blinks
            }
        return stats
    
    def get_antispoofing_info(self) -> Dict:
        """
        Get anti-spoofing configuration information.
        
        Returns:
            Dictionary with anti-spoofing settings
        """
        if not self.enable_antispoofing or self.antispoofing_detector is None:
            return {
                'enabled': False,
                'mode': 'disabled',
                'security_level': 'disabled',
                'reject_on_spoof': False
            }
        
        return {
            'enabled': True,
            'mode': self.antispoofing_mode,
            'detector_type': 'UltimateAntiSpoof',
            'security_level': SecurityLevel.STRICT.value if self.antispoofing_mode == "high_security" 
                            else SecurityLevel.BALANCED.value if self.antispoofing_mode == "basic"
                            else "disabled",
            'reject_on_spoof': self.reject_on_spoof
        }

    def __repr__(self) -> str:
        """Developer-friendly string representation."""
        status = "trained" if self.is_trained else "untrained"
        people_count = len(self.label_to_name)
        antispoofing_status = f", antispoofing={self.antispoofing_mode}" if self.enable_antispoofing else ""
        return (
            f"FaceRecognizer(status={status}, people={people_count}, "
            f"threshold={self.threshold}, enhanced={self.use_enhanced_preprocessing}{antispoofing_status})"
        )

    def __str__(self) -> str:
        """User-friendly string representation."""
        if not self.is_trained:
            return "FaceRecognizer (not trained)"
        
        people = self.get_known_people()
        people_str = ", ".join(people[:3])
        if len(people) > 3:
            people_str += f", ... ({len(people)} total)"
        
        return f"FaceRecognizer: {len(people)} people trained ({people_str})"