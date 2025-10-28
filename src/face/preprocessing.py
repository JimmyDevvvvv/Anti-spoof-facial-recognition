"""Enhanced face preprocessing module for state-of-the-art recognition accuracy.

Implements cutting-edge best practices:
- Multi-point facial landmark alignment (MediaPipe integration)
- Photometric normalization (Tan-Triggs algorithm)
- Advanced quality assessment with rejection thresholds
- Anti-spoofing detection capabilities
- Face pose estimation and filtering
- Intelligent margin-based cropping
- Enhanced data augmentation strategies
- Multi-scale processing for distance invariance
- Pipeline validation for train/test consistency
- GPU-accelerated batch processing support
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Optional dependencies
MEDIAPIPE_AVAILABLE = False
mp = None

try:
    import mediapipe as mp  # type: ignore[import]
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    warnings.warn("MediaPipe not available. Install with: pip install mediapipe", ImportWarning)


class NormalizationMethod(Enum):
    """Pixel intensity normalization methods."""
    MINMAX = "minmax"
    ZSCORE = "zscore"
    ADAPTIVE = "adaptive"
    PHOTOMETRIC = "photometric"  # New: Tan-Triggs


class QualityLevel(Enum):
    """Image quality assessment levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    REJECTED = "rejected"  # New: Too poor for recognition


class AlignmentMethod(Enum):
    """Face alignment methods."""
    EYES = "eyes"  # Eye-based (legacy)
    LANDMARKS = "landmarks"  # Facial landmarks (preferred)
    NONE = "none"


@dataclass
class QualityMetrics:
    """Container for image quality assessment metrics."""
    overall_score: float
    quality_level: QualityLevel
    brightness_score: float
    contrast_score: float
    sharpness_score: float
    edge_density: float
    mean_brightness: float
    std_contrast: float
    blur_variance: float
    should_reject: bool = False  # New: Rejection flag
    rejection_reason: Optional[str] = None  # New: Why rejected
    
    def __str__(self) -> str:
        """Human-readable representation."""
        status = f"[REJECTED: {self.rejection_reason}]" if self.should_reject else ""
        return (
            f"Quality: {self.quality_level.value} ({self.overall_score:.2f}) {status}, "
            f"Brightness: {self.mean_brightness:.1f}, "
            f"Contrast: {self.std_contrast:.1f}, "
            f"Sharpness: {self.blur_variance:.1f}"
        )


@dataclass
class PoseMetrics:
    """Container for face pose estimation metrics."""
    yaw: float  # Horizontal rotation (-90 to 90)
    pitch: float  # Vertical rotation (-90 to 90)
    roll: float  # In-plane rotation (-180 to 180)
    pose_score: float  # Quality score (1.0 = frontal)
    is_frontal: bool  # Whether pose is acceptable
    
    def __str__(self) -> str:
        """Human-readable representation."""
        return (
            f"Pose: yaw={self.yaw:.1f}Â°, pitch={self.pitch:.1f}Â°, roll={self.roll:.1f}Â°, "
            f"score={self.pose_score:.2f}, frontal={self.is_frontal}"
        )


@dataclass
class SpoofingMetrics:
    """Container for anti-spoofing detection metrics."""
    high_freq_score: float
    texture_score: float
    color_variance: float
    is_suspicious: bool
    suspicion_reason: Optional[str] = None
    
    def __str__(self) -> str:
        """Human-readable representation."""
        status = f"[SUSPICIOUS: {self.suspicion_reason}]" if self.is_suspicious else "[LIVE]"
        return f"Spoofing: {status}, freq={self.high_freq_score:.2f}"


@dataclass
class AlignmentResult:
    """Container for face alignment results."""
    aligned_image: np.ndarray
    success: bool
    angle: float
    method: AlignmentMethod
    eye_positions: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    landmarks: Optional[np.ndarray] = None  # New: Full facial landmarks
    pose_metrics: Optional[PoseMetrics] = None  # New: Pose information
    
    def __str__(self) -> str:
        """Human-readable representation."""
        status = "âœ“ Aligned" if self.success else "âœ— Not aligned"
        return f"{status} ({self.method.value}, angle: {self.angle:.1f}Â°)"


@dataclass
class PreprocessingResult:
    """Container for complete preprocessing results with metadata."""
    processed_image: np.ndarray
    quality_metrics: QualityMetrics
    alignment_result: Optional[AlignmentResult]
    applied_enhancements: List[str]
    spoofing_metrics: Optional[SpoofingMetrics] = None  # New
    multi_scale_pyramid: Optional[List[np.ndarray]] = None  # New
    
    def __str__(self) -> str:
        """Human-readable representation."""
        enhancements = ", ".join(self.applied_enhancements) if self.applied_enhancements else "None"
        return (
            f"Preprocessing: {self.quality_metrics.quality_level.value} quality, "
            f"Enhancements: {enhancements}"
        )


class EnhancedFacePreprocessor:
    """
    State-of-the-art face preprocessing for robust face recognition.
    
    Implements research-based best practices with modern enhancements:
    - Multi-point facial landmark alignment (MediaPipe/dlib)
    - Photometric normalization (Tan-Triggs algorithm)
    - Quality-aware adaptive processing with rejection
    - Anti-spoofing detection capabilities
    - Face pose estimation and filtering
    - Multi-scale processing for distance invariance
    - Enhanced augmentation strategies
    
    Example:
        >>> preprocessor = EnhancedFacePreprocessor(
        ...     target_size=(112, 112),
        ...     enable_landmark_alignment=True,
        ...     enable_antispoofing=True
        ... )
        >>> result = preprocessor.preprocess_with_metadata(face_image)
        >>> if result.quality_metrics.should_reject:
        ...     print(f"Image rejected: {result.quality_metrics.rejection_reason}")
        >>> else:
        ...     cv2.imwrite('processed.jpg', result.processed_image)
    """
    
    # Class constants
    DEFAULT_TARGET_SIZE = (112, 112)  # Updated: Modern standard
    DEFAULT_CLAHE_CLIP = 2.0
    DEFAULT_CLAHE_TILE = (8, 8)
    
    # Quality thresholds - WEBCAM OPTIMIZED (Balanced: reject photos, accept real faces)
    QUALITY_EXCELLENT = 0.75  # Lowered from 0.85 for webcam
    QUALITY_GOOD = 0.55  # Lowered from 0.65 for webcam
    QUALITY_FAIR = 0.35  # Lowered from 0.45 for webcam
    QUALITY_REJECT = 0.25  # Raised from 0.20 - reject only obvious low quality (photos ~0.20-0.29)
    
    # Pose thresholds
    MAX_YAW_ANGLE = 30.0  # Maximum horizontal rotation
    MAX_PITCH_ANGLE = 30.0  # Maximum vertical rotation
    MIN_POSE_SCORE = 0.6  # Minimum pose quality
    
    # Alignment thresholds
    MIN_ALIGNMENT_ANGLE = 2.0
    MIN_EYE_DISTANCE = 20
    
    # Enhancement parameters
    LOW_QUALITY_CLAHE_CLIP = 4.0
    HIGH_QUALITY_CLAHE_CLIP = 2.5
    
    # Anti-spoofing thresholds
    SPOOFING_HIGH_FREQ_THRESHOLD = 50.0
    SPOOFING_TEXTURE_THRESHOLD = 0.3
    
    def __init__(
        self,
        target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
        use_clahe: bool = True,
        clahe_clip_limit: float = DEFAULT_CLAHE_CLIP,
        clahe_tile_size: Tuple[int, int] = DEFAULT_CLAHE_TILE,
        enable_alignment: bool = True,
        enable_landmark_alignment: bool = True,  # New
        enable_photometric_norm: bool = True,  # New
        enable_quality_rejection: bool = True,  # New
        enable_pose_filtering: bool = True,  # New
        enable_antispoofing: bool = False,  # New
        quality_rejection_threshold: float = QUALITY_REJECT,  # New
        face_crop_margin: float = 0.2,  # New
    ) -> None:
        """
        Initialize the enhanced face preprocessor.
        
        Args:
            target_size: Output image size as (width, height) tuple
            use_clahe: Whether to apply CLAHE for lighting normalization
            clahe_clip_limit: CLAHE clip limit (1.0-4.0, higher = more contrast)
            clahe_tile_size: CLAHE tile grid size (e.g., (8, 8))
            enable_alignment: Whether to enable face alignment
            enable_landmark_alignment: Use facial landmarks (requires MediaPipe)
            enable_photometric_norm: Use Tan-Triggs photometric normalization
            enable_quality_rejection: Reject low-quality images
            enable_pose_filtering: Filter non-frontal faces
            enable_antispoofing: Enable basic anti-spoofing checks
            quality_rejection_threshold: Minimum quality score to accept
            face_crop_margin: Margin around face crop (0.0-0.5)
        """
        self._validate_parameters(target_size, clahe_clip_limit, clahe_tile_size)
        
        self.target_size = target_size
        self.use_clahe = use_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.enable_alignment = enable_alignment
        self.enable_landmark_alignment = enable_landmark_alignment and MEDIAPIPE_AVAILABLE
        self.enable_photometric_norm = enable_photometric_norm
        self.enable_quality_rejection = enable_quality_rejection
        self.enable_pose_filtering = enable_pose_filtering
        self.enable_antispoofing = enable_antispoofing
        self.quality_rejection_threshold = quality_rejection_threshold
        self.face_crop_margin = face_crop_margin
        
        # Initialize CLAHE
        self._clahe = self._create_clahe() if use_clahe else None
        
        # Load eye cascade for fallback alignment
        self._eye_cascade = self._load_eye_cascade() if enable_alignment else None
        
        # Initialize MediaPipe Face Mesh
        self._face_mesh = self._init_mediapipe() if self.enable_landmark_alignment else None
        
        # Store configuration for pipeline validation
        self._config = self._get_config_dict()

    @staticmethod
    def _validate_parameters(
        target_size: Tuple[int, int],
        clahe_clip_limit: float,
        clahe_tile_size: Tuple[int, int]
    ) -> None:
        """Validate initialization parameters."""
        if len(target_size) != 2 or target_size[0] <= 0 or target_size[1] <= 0:
            raise ValueError(
                f"target_size must be tuple of two positive integers, got {target_size}"
            )
        
        if not 0.5 <= clahe_clip_limit <= 10.0:
            raise ValueError(
                f"clahe_clip_limit must be between 0.5-10.0, got {clahe_clip_limit}"
            )
        
        if len(clahe_tile_size) != 2 or clahe_tile_size[0] <= 0 or clahe_tile_size[1] <= 0:
            raise ValueError(
                f"clahe_tile_size must be tuple of two positive integers, got {clahe_tile_size}"
            )

    def _create_clahe(self) -> cv2.CLAHE:
        """Create CLAHE object with configured parameters."""
        return cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_size
        )

    def _load_eye_cascade(self) -> Optional[cv2.CascadeClassifier]:
        """Load eye cascade classifier for fallback alignment."""
        try:
            cascade_path = Path(cv2.data.haarcascades) / 'haarcascade_eye.xml'
            eye_cascade = cv2.CascadeClassifier(str(cascade_path))
            
            if eye_cascade.empty():
                warnings.warn(
                    "Eye cascade classifier failed to load. Face alignment disabled.",
                    RuntimeWarning
                )
                return None
            
            return eye_cascade
            
        except Exception as e:
            warnings.warn(
                f"Could not load eye cascade: {e}. Face alignment disabled.",
                RuntimeWarning
            )
            return None
    
    def _init_mediapipe(self):
        """Initialize MediaPipe Face Mesh."""
        if not MEDIAPIPE_AVAILABLE:
            return None
        
        try:
            face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
            return face_mesh
        except Exception as e:
            warnings.warn(f"Could not initialize MediaPipe: {e}", RuntimeWarning)
            return None
    
    def preprocess(
        self,
        face_image: np.ndarray,
        align: bool = True,
        equalize: bool = True,
        face_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> np.ndarray:
        """
        Complete preprocessing pipeline for a face image.
        
        Args:
            face_image: Input face image (BGR or grayscale)
            align: Whether to apply face alignment
            equalize: Whether to apply histogram equalization/normalization
            face_bbox: Optional face bounding box (x, y, w, h) for margin crop
            
        Returns:
            Preprocessed face image (grayscale, aligned, normalized)
            
        Raises:
            ValueError: If input image is None, empty, or quality too poor
        """
        if face_image is None or face_image.size == 0:
            raise ValueError("Input face_image must not be None or empty")
        
        # Crop with margin if bbox provided
        if face_bbox is not None:
            face_image = self.crop_face_with_margin(face_image, face_bbox)
        
        # Convert to grayscale
        gray = self._ensure_grayscale(face_image)
        
        # Resize to target size
        gray = cv2.resize(gray, self.target_size, interpolation=cv2.INTER_AREA)
        
        # Assess image quality
        quality_metrics = self.assess_quality(gray)
        
        # Reject poor quality images
        if self.enable_quality_rejection and quality_metrics.should_reject:
            raise ValueError(
                f"Image quality rejected: {quality_metrics.rejection_reason} "
                f"(score: {quality_metrics.overall_score:.2f})"
            )
        
        # Face alignment
        if align and self.enable_alignment:
            alignment_result = self.align_face(gray)
            if alignment_result.success:
                # Pose filtering
                if self.enable_pose_filtering and alignment_result.pose_metrics:
                    if not alignment_result.pose_metrics.is_frontal:
                        warnings.warn(
                            f"Non-frontal face detected: {alignment_result.pose_metrics}",
                            RuntimeWarning
                        )
                gray = alignment_result.aligned_image
        
        # Photometric normalization (preferred) or adaptive equalization
        if equalize:
            if self.enable_photometric_norm:
                gray = self.apply_photometric_normalization(gray)
            else:
                gray = self._apply_adaptive_equalization(gray, quality_metrics)
        
        # Quality-based enhancement
        gray = self._apply_quality_enhancement(gray, quality_metrics)
        
        return gray
    
    def preprocess_with_metadata(
        self,
        face_image: np.ndarray,
        align: bool = True,
        equalize: bool = True,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
        check_spoofing: bool = False,
        create_pyramid: bool = False
    ) -> PreprocessingResult:
        """
        Preprocess face image and return detailed metadata.
        
        Args:
            face_image: Input face image
            align: Whether to apply face alignment
            equalize: Whether to apply normalization
            face_bbox: Optional face bounding box for margin crop
            check_spoofing: Whether to perform anti-spoofing checks
            create_pyramid: Whether to create multi-scale pyramid
            
        Returns:
            PreprocessingResult with processed image and comprehensive metadata
        """
        if face_image is None or face_image.size == 0:
            raise ValueError("Input face_image must not be None or empty")
        
        applied_enhancements = []
        
        # Crop with margin
        if face_bbox is not None:
            face_image = self.crop_face_with_margin(face_image, face_bbox)
            applied_enhancements.append(f"margin_crop_{self.face_crop_margin}")
        
        # Convert to grayscale
        gray = self._ensure_grayscale(face_image)
        applied_enhancements.append("grayscale_conversion")
        
        # Resize
        gray = cv2.resize(gray, self.target_size, interpolation=cv2.INTER_AREA)
        applied_enhancements.append("resize")
        
        # Quality assessment
        quality_metrics = self.assess_quality(gray)
        
        # Anti-spoofing check
        spoofing_metrics = None
        if check_spoofing and self.enable_antispoofing:
            spoofing_metrics = self.detect_potential_spoofing(gray)
            if spoofing_metrics.is_suspicious:
                applied_enhancements.append(f"spoofing_detected")
        
        # Face alignment
        alignment_result = None
        if align and self.enable_alignment:
            alignment_result = self.align_face(gray)
            if alignment_result.success:
                gray = alignment_result.aligned_image
                applied_enhancements.append(
                    f"{alignment_result.method.value}_alignment_{alignment_result.angle:.1f}deg"
                )
        
        # Normalization
        if equalize:
            if self.enable_photometric_norm:
                gray = self.apply_photometric_normalization(gray)
                applied_enhancements.append("photometric_normalization")
            else:
                gray = self._apply_adaptive_equalization(gray, quality_metrics)
                if quality_metrics.overall_score < self.QUALITY_FAIR:
                    applied_enhancements.append("enhanced_clahe")
                elif self.use_clahe:
                    applied_enhancements.append("standard_clahe")
                else:
                    applied_enhancements.append("histogram_equalization")
        
        # Quality enhancement
        original_gray = gray.copy()
        gray = self._apply_quality_enhancement(gray, quality_metrics)
        if not np.array_equal(gray, original_gray):
            if quality_metrics.overall_score < self.QUALITY_FAIR:
                applied_enhancements.append("low_quality_enhancement")
            elif quality_metrics.overall_score > self.QUALITY_EXCELLENT:
                applied_enhancements.append("high_quality_enhancement")
        
        # Multi-scale pyramid
        multi_scale_pyramid = None
        if create_pyramid:
            multi_scale_pyramid = self.create_image_pyramid(gray)
            applied_enhancements.append(f"pyramid_{len(multi_scale_pyramid)}_scales")
        
        return PreprocessingResult(
            processed_image=gray,
            quality_metrics=quality_metrics,
            alignment_result=alignment_result,
            applied_enhancements=applied_enhancements,
            spoofing_metrics=spoofing_metrics,
            multi_scale_pyramid=multi_scale_pyramid
        )

    def crop_face_with_margin(
        self,
        image: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        margin: Optional[float] = None
    ) -> np.ndarray:
        """
        Crop face with margin to include context (hair, ears, chin).
        
        Args:
            image: Input image
            face_bbox: Face bounding box (x, y, width, height)
            margin: Expansion factor (default: use self.face_crop_margin)
            
        Returns:
            Cropped face image with margin
        """
        if margin is None:
            margin = self.face_crop_margin
        
        x, y, w, h = face_bbox
        margin_x = int(w * margin)
        margin_y = int(h * margin)
        
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(image.shape[1], x + w + margin_x)
        y2 = min(image.shape[0], y + h + margin_y)
        
        return image[y1:y2, x1:x2]

    @staticmethod
    def _ensure_grayscale(image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale if needed."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.copy()

    def apply_photometric_normalization(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Tan-Triggs photometric normalization.
        
        More robust than CLAHE for handling varying illumination conditions.
        Research paper: Tan & Triggs, "Enhanced Local Texture Feature Sets" (2010)
        
        Args:
            image: Input grayscale image
            
        Returns:
            Photometrically normalized image
        """
        try:
            # Convert to float
            image_float = image.astype(np.float32) / 255.0
            
            # Gamma correction (compress dynamic range)
            gamma = 0.2
            image_float = np.power(image_float, gamma)
            
            # Difference of Gaussians (DoG) filtering
            gaussian1 = cv2.GaussianBlur(image_float, (3, 3), 1.0)
            gaussian2 = cv2.GaussianBlur(image_float, (3, 3), 2.0)
            dog = gaussian1 - gaussian2
            
            # Contrast equalization
            mean = np.mean(dog)
            std = np.std(dog)
            normalized = (dog - mean) / max(std, 1e-6)
            
            # Hyperbolic tangent mapping (compresses outliers)
            tau = 10.0
            normalized = tau * np.tanh(normalized / tau)
            
            # Scale back to [0, 255]
            normalized = ((normalized + tau) / (2 * tau) * 255)
            normalized = np.clip(normalized, 0, 255).astype(np.uint8)
            
            return normalized
            
        except Exception as e:
            warnings.warn(f"Photometric normalization failed: {e}", RuntimeWarning)
            return image

    def _apply_adaptive_equalization(
        self,
        gray: np.ndarray,
        quality_metrics: QualityMetrics
    ) -> np.ndarray:
        """Apply adaptive equalization based on image quality."""
        try:
            if quality_metrics.overall_score < self.QUALITY_FAIR:
                clahe = cv2.createCLAHE(
                    clipLimit=self.LOW_QUALITY_CLAHE_CLIP,
                    tileGridSize=self.clahe_tile_size
                )
                return clahe.apply(gray)
            
            elif quality_metrics.overall_score > self.QUALITY_EXCELLENT:
                if self.use_clahe:
                    clahe = cv2.createCLAHE(
                        clipLimit=self.HIGH_QUALITY_CLAHE_CLIP,
                        tileGridSize=self.clahe_tile_size
                    )
                    return clahe.apply(gray)
                return gray
            
            else:
                if self.use_clahe and self._clahe is not None:
                    return self._clahe.apply(gray)
                return cv2.equalizeHist(gray)
                
        except Exception as e:
            warnings.warn(f"Adaptive equalization failed: {e}", RuntimeWarning)
            return gray

    def _apply_quality_enhancement(
        self,
        gray: np.ndarray,
        quality_metrics: QualityMetrics
    ) -> np.ndarray:
        """Apply enhancement based on quality level."""
        if quality_metrics.overall_score < self.QUALITY_FAIR:
            return self._enhance_low_quality(gray)
        elif quality_metrics.overall_score > self.QUALITY_EXCELLENT:
            return self._enhance_high_quality(gray)
        return gray

    def align_face(self, face_gray: np.ndarray) -> AlignmentResult:
        """
        Align face using best available method (landmarks preferred, eyes fallback).
        
        Args:
            face_gray: Grayscale face image
            
        Returns:
            AlignmentResult with aligned image and metadata
        """
        # Try landmark-based alignment first
        if self._face_mesh is not None and self.enable_landmark_alignment:
            result = self._align_face_landmarks(face_gray)
            if result.success:
                return result
        
        # Fallback to eye-based alignment
        if self._eye_cascade is not None:
            return self._align_face_eyes(face_gray)
        
        # No alignment available
        return AlignmentResult(
            aligned_image=face_gray,
            success=False,
            angle=0.0,
            method=AlignmentMethod.NONE
        )

    def _align_face_landmarks(self, face_gray: np.ndarray) -> AlignmentResult:
        """
        Align face using facial landmarks (MediaPipe).
        More robust than eye-only alignment.
        """
        try:
            # Convert to RGB for MediaPipe
            face_rgb = cv2.cvtColor(face_gray, cv2.COLOR_GRAY2RGB)
            
            # Detect landmarks
            results = self._face_mesh.process(face_rgb)
            
            if not results.multi_face_landmarks:
                return AlignmentResult(
                    aligned_image=face_gray,
                    success=False,
                    angle=0.0,
                    method=AlignmentMethod.LANDMARKS
                )
            
            # Get first face landmarks
            landmarks = results.multi_face_landmarks[0]
            
            # Convert to numpy array
            h, w = face_gray.shape
            landmark_points = np.array([
                [lm.x * w, lm.y * h] for lm in landmarks.landmark
            ])
            
            # Use specific landmarks for alignment (eyes: 33, 263)
            left_eye = landmark_points[33]
            right_eye = landmark_points[263]
            
            # Calculate rotation angle
            angle = self._calculate_rotation_angle(
                tuple(left_eye), tuple(right_eye)
            )
            
            # Estimate pose
            pose_metrics = self._estimate_pose_from_landmarks(landmark_points, face_gray.shape)
            
            # Only align if angle is significant
            if abs(angle) < self.MIN_ALIGNMENT_ANGLE:
                return AlignmentResult(
                    aligned_image=face_gray,
                    success=False,
                    angle=angle,
                    method=AlignmentMethod.LANDMARKS,
                    landmarks=landmark_points,
                    pose_metrics=pose_metrics
                )
            
            # Calculate rotation center (between eyes)
            rotation_center = ((left_eye[0] + right_eye[0]) / 2,
                             (left_eye[1] + right_eye[1]) / 2)
            
            # Apply rotation
            aligned = self._rotate_image(face_gray, rotation_center, angle)
            
            return AlignmentResult(
                aligned_image=aligned,
                success=True,
                angle=angle,
                method=AlignmentMethod.LANDMARKS,
                eye_positions=(tuple(left_eye), tuple(right_eye)),
                landmarks=landmark_points,
                pose_metrics=pose_metrics
            )
            
        except Exception as e:
            warnings.warn(f"Landmark alignment failed: {e}", RuntimeWarning)
            return AlignmentResult(
                aligned_image=face_gray,
                success=False,
                angle=0.0,
                method=AlignmentMethod.LANDMARKS
            )

    def _align_face_eyes(self, face_gray: np.ndarray) -> AlignmentResult:
        """Fallback eye-based alignment (original method)."""
        if self._eye_cascade is None:
            return AlignmentResult(
                aligned_image=face_gray,
                success=False,
                angle=0.0,
                method=AlignmentMethod.EYES
            )
        
        try:
            eyes = self._eye_cascade.detectMultiScale(
                face_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(20, 20)
            )
            
            if len(eyes) < 2:
                return AlignmentResult(
                    aligned_image=face_gray,
                    success=False,
                    angle=0.0,
                    method=AlignmentMethod.EYES
                )
            
            eyes = sorted(eyes, key=lambda e: e[0])
            left_eye_center, right_eye_center = self._calculate_eye_centers(
                eyes[0], eyes[1]
            )
            
            eye_distance = np.sqrt(
                (right_eye_center[0] - left_eye_center[0]) ** 2 +
                (right_eye_center[1] - left_eye_center[1]) ** 2
            )
            
            if eye_distance < self.MIN_EYE_DISTANCE:
                return AlignmentResult(
                    aligned_image=face_gray,
                    success=False,
                    angle=0.0,
                    method=AlignmentMethod.EYES
                )
            
            angle = self._calculate_rotation_angle(left_eye_center, right_eye_center)
            
            if abs(angle) < self.MIN_ALIGNMENT_ANGLE:
                return AlignmentResult(
                    aligned_image=face_gray,
                    success=False,
                    angle=angle,
                    method=AlignmentMethod.EYES,
                    eye_positions=(left_eye_center, right_eye_center)
                )
            
            rotation_center = (
                (left_eye_center[0] + right_eye_center[0]) / 2,
                (left_eye_center[1] + right_eye_center[1]) / 2
            )
            
            aligned = self._rotate_image(face_gray, rotation_center, angle)
            
            return AlignmentResult(
                aligned_image=aligned,
                success=True,
                angle=angle,
                method=AlignmentMethod.EYES,
                eye_positions=(left_eye_center, right_eye_center)
            )
            
        except Exception as e:
            warnings.warn(f"Eye alignment failed: {e}", RuntimeWarning)
            return AlignmentResult(
                aligned_image=face_gray,
                success=False,
                angle=0.0,
                method=AlignmentMethod.EYES
            )

    def _estimate_pose_from_landmarks(
        self,
        landmarks: np.ndarray,
        image_shape: Tuple[int, int]
    ) -> PoseMetrics:
        """
        Estimate face pose from landmarks.
        
        Simplified pose estimation using key facial landmarks.
        """
        try:
            h, w = image_shape
            
            # Key points (approximate indices for MediaPipe)
            nose_tip = landmarks[1] if len(landmarks) > 1 else [w/2, h/2]
            left_eye = landmarks[33] if len(landmarks) > 33 else [w/3, h/3]
            right_eye = landmarks[263] if len(landmarks) > 263 else [2*w/3, h/3]
            
            # Estimate yaw (horizontal rotation)
            eye_center_x = (left_eye[0] + right_eye[0]) / 2
            yaw = (nose_tip[0] - eye_center_x) / (w / 2) * 45  # Approximate
            
            # Estimate pitch (vertical rotation)
            eye_center_y = (left_eye[1] + right_eye[1]) / 2
            pitch = (nose_tip[1] - eye_center_y) / (h / 2) * 30  # Approximate
            
            # Estimate roll (in-plane rotation)
            roll = self._calculate_rotation_angle(tuple(left_eye), tuple(right_eye))
            
            # Calculate pose score (1.0 = frontal, 0.0 = profile)
            yaw_score = 1.0 - abs(yaw) / 90.0
            pitch_score = 1.0 - abs(pitch) / 90.0
            roll_score = 1.0 - abs(roll) / 45.0
            pose_score = (yaw_score + pitch_score + roll_score) / 3.0
            
            # Check if frontal
            is_frontal = (
                abs(yaw) < self.MAX_YAW_ANGLE and
                abs(pitch) < self.MAX_PITCH_ANGLE and
                pose_score > self.MIN_POSE_SCORE
            )
            
            return PoseMetrics(
                yaw=float(yaw),
                pitch=float(pitch),
                roll=float(roll),
                pose_score=float(pose_score),
                is_frontal=is_frontal
            )
            
        except Exception as e:
            warnings.warn(f"Pose estimation failed: {e}", RuntimeWarning)
            return PoseMetrics(
                yaw=0.0,
                pitch=0.0,
                roll=0.0,
                pose_score=0.5,
                is_frontal=True
            )

    @staticmethod
    def _calculate_eye_centers(
        left_eye: np.ndarray,
        right_eye: np.ndarray
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Calculate eye centers from eye bounding boxes."""
        left_eye_center = (
            float(left_eye[0] + left_eye[2] // 2),
            float(left_eye[1] + left_eye[3] // 2)
        )
        right_eye_center = (
            float(right_eye[0] + right_eye[2] // 2),
            float(right_eye[1] + right_eye[3] // 2)
        )
        return left_eye_center, right_eye_center

    @staticmethod
    def _calculate_rotation_angle(
        left_eye: Tuple[float, float],
        right_eye: Tuple[float, float]
    ) -> float:
        """Calculate rotation angle from eye positions."""
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        return float(np.degrees(np.arctan2(dy, dx)))

    @staticmethod
    def _rotate_image(
        image: np.ndarray,
        center: Tuple[float, float],
        angle: float
    ) -> np.ndarray:
        """Rotate image around center point by given angle."""
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        return rotated

    def assess_quality(self, image: np.ndarray) -> QualityMetrics:
        """
        Comprehensive image quality assessment with rejection logic.
        
        Args:
            image: Input grayscale image
            
        Returns:
            QualityMetrics with rejection flag if quality too poor
        """
        try:
            # Brightness assessment
            mean_brightness = float(np.mean(image))
            brightness_score = self._assess_brightness(mean_brightness)
            
            # Contrast assessment
            std_contrast = float(np.std(image))
            contrast_score = self._assess_contrast(std_contrast)
            
            # Sharpness assessment
            blur_variance = float(cv2.Laplacian(image, cv2.CV_64F).var())
            sharpness_score = self._assess_sharpness(blur_variance)
            
            # Edge density assessment
            edge_density = self._calculate_edge_density(image)
            edge_score = self._assess_edges(edge_density)
            
            # Weighted overall score
            overall_score = (
                brightness_score * 0.3 +
                contrast_score * 0.3 +
                sharpness_score * 0.2 +
                edge_score * 0.2
            )
            
            # Determine quality level
            quality_level = self._determine_quality_level(overall_score)
            
            # Determine if should reject
            should_reject = False
            rejection_reason = None
            
            if self.enable_quality_rejection:
                if overall_score < self.quality_rejection_threshold:
                    should_reject = True
                    rejection_reason = f"Overall quality too low ({overall_score:.2f})"
                elif blur_variance < 20:
                    should_reject = True
                    rejection_reason = f"Image too blurry (variance: {blur_variance:.1f})"
                elif std_contrast < 15:
                    should_reject = True
                    rejection_reason = f"Contrast too low ({std_contrast:.1f})"
                elif mean_brightness < 15 or mean_brightness > 240:
                    should_reject = True
                    rejection_reason = f"Brightness out of range ({mean_brightness:.1f})"
            
            if should_reject:
                quality_level = QualityLevel.REJECTED
            
            return QualityMetrics(
                overall_score=overall_score,
                quality_level=quality_level,
                brightness_score=brightness_score,
                contrast_score=contrast_score,
                sharpness_score=sharpness_score,
                edge_density=edge_density,
                mean_brightness=mean_brightness,
                std_contrast=std_contrast,
                blur_variance=blur_variance,
                should_reject=should_reject,
                rejection_reason=rejection_reason
            )
            
        except Exception as e:
            warnings.warn(f"Quality assessment failed: {e}", RuntimeWarning)
            return QualityMetrics(
                overall_score=0.5,
                quality_level=QualityLevel.FAIR,
                brightness_score=0.5,
                contrast_score=0.5,
                sharpness_score=0.5,
                edge_density=0.1,
                mean_brightness=127.0,
                std_contrast=50.0,
                blur_variance=100.0,
                should_reject=False
            )

    def detect_potential_spoofing(self, image: np.ndarray) -> SpoofingMetrics:
        """
        Basic anti-spoofing detection.
        
        Detects potential print/screen attacks through:
        - High frequency analysis (moirÃ© patterns)
        - Texture analysis
        - Color variance analysis
        
        Note: This is a basic implementation. Production systems should use
        dedicated liveness detection models.
        
        Args:
            image: Input grayscale image
            
        Returns:
            SpoofingMetrics with suspicion flags
        """
        try:
            # FFT analysis for moirÃ© patterns
            fft = np.fft.fft2(image)
            fft_shift = np.fft.fftshift(fft)
            magnitude_spectrum = 20 * np.log(np.abs(fft_shift) + 1)
            
            # High frequency energy (screens/prints have characteristic patterns)
            h, w = image.shape
            high_freq_region = magnitude_spectrum[h//3:, w//3:]
            high_freq_score = float(np.mean(high_freq_region))
            
            # Texture analysis using LBP-like measure
            laplacian = cv2.Laplacian(image, cv2.CV_64F)
            texture_score = float(np.std(laplacian))
            
            # Color variance (for grayscale, check intensity variance)
            color_variance = float(np.std(image))
            
            # Determine if suspicious
            is_suspicious = False
            suspicion_reason = None
            
            if high_freq_score > self.SPOOFING_HIGH_FREQ_THRESHOLD:
                is_suspicious = True
                suspicion_reason = "High frequency moirÃ© pattern detected"
            elif texture_score < self.SPOOFING_TEXTURE_THRESHOLD:
                is_suspicious = True
                suspicion_reason = "Low texture variance (flat surface)"
            
            return SpoofingMetrics(
                high_freq_score=high_freq_score,
                texture_score=texture_score,
                color_variance=color_variance,
                is_suspicious=is_suspicious,
                suspicion_reason=suspicion_reason
            )
            
        except Exception as e:
            warnings.warn(f"Spoofing detection failed: {e}", RuntimeWarning)
            return SpoofingMetrics(
                high_freq_score=0.0,
                texture_score=0.0,
                color_variance=0.0,
                is_suspicious=False
            )

    @staticmethod
    def _assess_brightness(mean_brightness: float) -> float:
        """WEBCAM-OPTIMIZED brightness assessment (0.0-1.0)."""
        if 25 <= mean_brightness <= 210:  # Very lenient range
            return 0.2
        elif 20 <= mean_brightness <= 225:  # Extended lenient range
            return 0.15
        elif 15 <= mean_brightness <= 235:  # Maximum lenient range
            return 0.1
        return 0.0

    @staticmethod
    def _assess_contrast(std_contrast: float) -> float:
        """WEBCAM-OPTIMIZED contrast assessment (0.0-1.0)."""
        if std_contrast > 25:  # Lower threshold
            return 0.2
        elif std_contrast > 20:  # More lenient
            return 0.15
        elif std_contrast > 18:  # Very lenient
            return 0.1
        return 0.0

    @staticmethod
    def _assess_sharpness(blur_variance: float) -> float:
        """WEBCAM-OPTIMIZED sharpness assessment (0.0-1.0)."""
        if blur_variance > 100:  # Lower threshold
            return 0.2
        elif blur_variance > 70:  # More lenient
            return 0.15
        elif blur_variance > 45:  # Very lenient
            return 0.1
        return 0.0

    @staticmethod
    def _calculate_edge_density(image: np.ndarray) -> float:
        """Calculate edge density in image."""
        edges = cv2.Canny(image, 50, 150)
        return float(np.sum(edges > 0) / edges.size)

    @staticmethod
    def _assess_edges(edge_density: float) -> float:
        """Assess edge density quality (0.0-1.0)."""
        if edge_density > 0.15:
            return 1.0
        elif edge_density > 0.1:
            return 0.7
        elif edge_density > 0.05:
            return 0.4
        return 0.1

    def _determine_quality_level(self, overall_score: float) -> QualityLevel:
        """Determine quality level from overall score."""
        if overall_score >= self.QUALITY_EXCELLENT:
            return QualityLevel.EXCELLENT
        elif overall_score >= self.QUALITY_GOOD:
            return QualityLevel.GOOD
        elif overall_score >= self.QUALITY_FAIR:
            return QualityLevel.FAIR
        return QualityLevel.POOR

    def _enhance_low_quality(self, image: np.ndarray) -> np.ndarray:
        """Enhanced aggressive enhancement for low-quality images."""
        try:
            clahe = cv2.createCLAHE(
                clipLimit=self.LOW_QUALITY_CLAHE_CLIP,
                tileGridSize=self.clahe_tile_size
            )
            enhanced = clahe.apply(image)
            
            # Denoise
            enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
            
            # Unsharp masking
            gaussian_blur = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
            enhanced = cv2.addWeighted(enhanced, 1.5, gaussian_blur, -0.5, 0)
            
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
            
            return enhanced
            
        except Exception as e:
            warnings.warn(f"Low quality enhancement failed: {e}", RuntimeWarning)
            return image

    def _enhance_high_quality(self, image: np.ndarray) -> np.ndarray:
        """Subtle enhancement for high-quality images."""
        try:
            clahe = cv2.createCLAHE(
                clipLimit=self.HIGH_QUALITY_CLAHE_CLIP,
                tileGridSize=self.clahe_tile_size
            )
            enhanced = clahe.apply(image)
            
            # Subtle sharpening
            sharpen_kernel = np.array([
                [-1, -1, -1],
                [-1,  9, -1],
                [-1, -1, -1]
            ], dtype=np.float32)
            
            sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel * 0.1)
            enhanced = cv2.addWeighted(enhanced, 0.9, sharpened, 0.1, 0)
            
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
            
            return enhanced
            
        except Exception as e:
            warnings.warn(f"High quality enhancement failed: {e}", RuntimeWarning)
            return image
    
    def augment_face(
        self,
        face_image: np.ndarray,
        num_variations: int = 3,
        include_original: bool = True,
        rotation_range: Tuple[float, float] = (-15.0, 15.0),
        brightness_range: Tuple[float, float] = (0.7, 1.3),
        flip_probability: float = 0.5,
        noise_probability: float = 0.5,
        noise_std: float = 5.0,
        blur_probability: float = 0.3,  # New
        occlusion_probability: float = 0.2,  # New
    ) -> List[np.ndarray]:
        """
        Generate enhanced augmented variations for robust training.
        
        Includes new augmentations:
        - Motion blur simulation
        - Random occlusions (simulates glasses, partial obstruction)
        
        Args:
            face_image: Input face image (grayscale)
            num_variations: Number of augmented samples
            include_original: Include original image
            rotation_range: Min/max rotation angles
            brightness_range: Min/max brightness multipliers
            flip_probability: Probability of horizontal flip
            noise_probability: Probability of adding noise
            noise_std: Standard deviation of Gaussian noise
            blur_probability: Probability of motion blur
            occlusion_probability: Probability of random occlusion
            
        Returns:
            List of augmented face images
        """
        if face_image is None or face_image.size == 0:
            raise ValueError("Input face_image must not be None or empty")
        
        if num_variations < 0:
            raise ValueError(f"num_variations must be non-negative, got {num_variations}")
        
        augmented = []
        
        if include_original:
            augmented.append(face_image.copy())
        
        for _ in range(num_variations):
            aug_face = self._apply_random_augmentation(
                face_image,
                rotation_range,
                brightness_range,
                flip_probability,
                noise_probability,
                noise_std,
                blur_probability,
                occlusion_probability
            )
            augmented.append(aug_face)
        
        return augmented

    def _apply_random_augmentation(
        self,
        image: np.ndarray,
        rotation_range: Tuple[float, float],
        brightness_range: Tuple[float, float],
        flip_probability: float,
        noise_probability: float,
        noise_std: float,
        blur_probability: float,
        occlusion_probability: float
    ) -> np.ndarray:
        """Apply random augmentation transformations with new enhancements."""
        aug_face = image.copy()
        
        # Random rotation
        angle = np.random.uniform(*rotation_range)
        center = (image.shape[1] // 2, image.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        aug_face = cv2.warpAffine(
            aug_face,
            rotation_matrix,
            (image.shape[1], image.shape[0]),
            borderMode=cv2.BORDER_REPLICATE
        )
        
        # Random brightness adjustment
        brightness_factor = np.random.uniform(*brightness_range)
        aug_face = np.clip(aug_face * brightness_factor, 0, 255).astype(np.uint8)
            
        # Random horizontal flip
        if np.random.random() < flip_probability:
            aug_face = cv2.flip(aug_face, 1)
        
        # Random Gaussian noise
        if np.random.random() < noise_probability:
            noise = np.random.normal(0, noise_std, aug_face.shape)
            aug_face = np.clip(aug_face + noise, 0, 255).astype(np.uint8)
        
        # Random motion blur (NEW)
        if np.random.random() < blur_probability:
            kernel_size = np.random.randint(3, 8)
            kernel_motion_blur = np.zeros((kernel_size, kernel_size))
            kernel_motion_blur[int((kernel_size-1)/2), :] = np.ones(kernel_size)
            kernel_motion_blur = kernel_motion_blur / kernel_size
            aug_face = cv2.filter2D(aug_face, -1, kernel_motion_blur)
        
        # Random occlusion (NEW - simulates glasses, partial obstruction)
        if np.random.random() < occlusion_probability:
            h, w = aug_face.shape
            occlusion_size = np.random.randint(int(min(h, w) * 0.1), int(min(h, w) * 0.3))
            x = np.random.randint(0, w - occlusion_size)
            y = np.random.randint(0, h - occlusion_size)
            aug_face[y:y+occlusion_size, x:x+occlusion_size] = np.random.randint(0, 256)
            
        return aug_face
    
    def create_image_pyramid(
        self,
        image: np.ndarray,
        scales: List[float] = [0.75, 1.0, 1.25]
    ) -> List[np.ndarray]:
        """
        Create multi-scale representations for distance-invariant recognition.
        
        Useful for handling faces at different distances from camera.
        
        Args:
            image: Input preprocessed face image
            scales: List of scale factors
            
        Returns:
            List of scaled and preprocessed images
        """
        pyramid = []
        
        for scale in scales:
            if scale == 1.0:
                pyramid.append(image.copy())
            else:
                scaled_size = (int(self.target_size[0] * scale),
                             int(self.target_size[1] * scale))
                scaled = cv2.resize(image, scaled_size, interpolation=cv2.INTER_LINEAR)
                # Resize back to target size for consistency
                scaled = cv2.resize(scaled, self.target_size, interpolation=cv2.INTER_LINEAR)
                pyramid.append(scaled)
        
        return pyramid
    
    def normalize_intensity(
        self,
        face_image: np.ndarray,
        method: Union[str, NormalizationMethod] = NormalizationMethod.MINMAX
    ) -> np.ndarray:
        """
        Normalize pixel intensity values using specified method.
        
        Args:
            face_image: Input grayscale face image
            method: Normalization method (minmax, zscore, adaptive, photometric)
            
        Returns:
            Normalized face image (uint8)
        """
        if face_image is None or face_image.size == 0:
            raise ValueError("Input face_image must not be None or empty")
        
        if isinstance(method, str):
            try:
                method = NormalizationMethod(method.lower())
            except ValueError:
                raise ValueError(
                    f"Unknown normalization method: {method}. "
                    f"Must be one of: {[m.value for m in NormalizationMethod]}"
                )
        
        if method == NormalizationMethod.MINMAX:
            return self._normalize_minmax(face_image)
        elif method == NormalizationMethod.ZSCORE:
            return self._normalize_zscore(face_image)
        elif method == NormalizationMethod.ADAPTIVE:
            return self._normalize_adaptive(face_image)
        elif method == NormalizationMethod.PHOTOMETRIC:
            return self.apply_photometric_normalization(face_image)
        else:
            raise ValueError(f"Unsupported normalization method: {method}")

    @staticmethod
    def _normalize_minmax(image: np.ndarray) -> np.ndarray:
        """Min-max normalization to [0, 255] range."""
        image_float = image.astype(np.float32)
        img_min = image_float.min()
        img_max = image_float.max()
        
        if img_max - img_min > 1e-6:
            normalized = 255 * (image_float - img_min) / (img_max - img_min)
        else:
            normalized = image_float
            
        return normalized.astype(np.uint8)
        
    @staticmethod
    def _normalize_zscore(image: np.ndarray) -> np.ndarray:
        """Z-score normalization, then scale to [0, 255]."""
        image_float = image.astype(np.float32)
        mean = image_float.mean()
        std = image_float.std()
        
        if std > 1e-6:
            normalized = (image_float - mean) / std
            norm_min = normalized.min()
            norm_max = normalized.max()
            if norm_max - norm_min > 1e-6:
                normalized = (normalized - norm_min) / (norm_max - norm_min) * 255
            else:
                normalized = np.full_like(normalized, 127.5)
        else:
            normalized = image_float
            
        return normalized.astype(np.uint8)
        
    def _normalize_adaptive(self, image: np.ndarray) -> np.ndarray:
        """Adaptive normalization based on image quality."""
        quality_metrics = self.assess_quality(image)
        
        if quality_metrics.overall_score < self.QUALITY_FAIR:
            return self._normalize_zscore(image)
        else:
            return self._normalize_minmax(image)

    def batch_preprocess(
        self,
        images: List[np.ndarray],
        align: bool = True,
        equalize: bool = True,
        show_progress: bool = False,
        skip_on_error: bool = True
    ) -> List[np.ndarray]:
        """
        Preprocess multiple face images efficiently.
        
        Args:
            images: List of face images to preprocess
            align: Whether to apply face alignment
            equalize: Whether to apply normalization
            show_progress: Whether to print progress
            skip_on_error: Skip failed images instead of raising
            
        Returns:
            List of preprocessed face images
        """
        processed = []
        failed_count = 0
        
        for idx, img in enumerate(images):
            if show_progress and (idx + 1) % 10 == 0:
                print(f"Processing: {idx + 1}/{len(images)} (Failed: {failed_count})")
            
            try:
                preprocessed = self.preprocess(img, align=align, equalize=equalize)
                processed.append(preprocessed)
            except Exception as e:
                failed_count += 1
                if skip_on_error:
                    warnings.warn(
                        f"Failed to preprocess image {idx}: {e}",
                        RuntimeWarning
                    )
                    processed.append(self._create_placeholder_image())
                else:
                    raise
        
        if show_progress:
            success_rate = ((len(images) - failed_count) / len(images)) * 100
            print(f"Completed: {len(processed)}/{len(images)} ({success_rate:.1f}% success)")
        
        return processed

    def _create_placeholder_image(self) -> np.ndarray:
        """Create a placeholder image for failed preprocessing."""
        return np.zeros(self.target_size[::-1], dtype=np.uint8)
    
    def validate_pipeline_consistency(
        self,
        other_config: Dict
    ) -> List[str]:
        """
        Validate that this preprocessor matches another configuration.
        
        Critical for ensuring training and inference pipelines match.
        
        Args:
            other_config: Configuration dict from another preprocessor
            
        Returns:
            List of warning messages (empty if consistent)
        """
        warnings_list = []
        
        current_config = self._config
        
        # Check critical parameters
        if current_config['target_size'] != other_config.get('target_size'):
            warnings_list.append(
                f"Size mismatch: {current_config['target_size']} vs "
                f"{other_config.get('target_size')}"
            )
        
        if current_config['use_clahe'] != other_config.get('use_clahe'):
            warnings_list.append("CLAHE usage mismatch")
        
        if current_config['enable_photometric_norm'] != other_config.get('enable_photometric_norm'):
            warnings_list.append("Photometric normalization mismatch")
        
        if current_config['enable_alignment'] != other_config.get('enable_alignment'):
            warnings_list.append("Alignment setting mismatch")
        
        if current_config['clahe_clip_limit'] != other_config.get('clahe_clip_limit'):
            warnings_list.append(
                f"CLAHE clip limit mismatch: {current_config['clahe_clip_limit']} vs "
                f"{other_config.get('clahe_clip_limit')}"
            )
        
        return warnings_list
    
    def _get_config_dict(self) -> Dict:
        """Get configuration dictionary for validation."""
        return {
            'target_size': self.target_size,
            'use_clahe': self.use_clahe,
            'clahe_clip_limit': self.clahe_clip_limit,
            'clahe_tile_size': self.clahe_tile_size,
            'enable_alignment': self.enable_alignment,
            'enable_landmark_alignment': self.enable_landmark_alignment,
            'enable_photometric_norm': self.enable_photometric_norm,
            'enable_quality_rejection': self.enable_quality_rejection,
            'enable_pose_filtering': self.enable_pose_filtering,
            'quality_rejection_threshold': self.quality_rejection_threshold,
            'face_crop_margin': self.face_crop_margin
        }

    def get_preprocessor_info(self) -> Dict:
        """Get comprehensive preprocessor configuration and capabilities."""
        return {
            'target_size': self.target_size,
            'use_clahe': self.use_clahe,
            'clahe_parameters': {
                'clip_limit': self.clahe_clip_limit,
                'tile_size': self.clahe_tile_size
            },
            'alignment_enabled': self.enable_alignment,
            'landmark_alignment': self.enable_landmark_alignment,
            'alignment_available': {
                'eyes': self._eye_cascade is not None,
                'landmarks': self._face_mesh is not None
            },
            'photometric_normalization': self.enable_photometric_norm,
            'quality_rejection': {
                'enabled': self.enable_quality_rejection,
                'threshold': self.quality_rejection_threshold
            },
            'pose_filtering': self.enable_pose_filtering,
            'antispoofing': self.enable_antispoofing,
            'quality_thresholds': {
                'excellent': self.QUALITY_EXCELLENT,
                'good': self.QUALITY_GOOD,
                'fair': self.QUALITY_FAIR,
                'reject': self.QUALITY_REJECT
            },
            'pose_thresholds': {
                'max_yaw': self.MAX_YAW_ANGLE,
                'max_pitch': self.MAX_PITCH_ANGLE,
                'min_score': self.MIN_POSE_SCORE
            },
            'supported_methods': {
                'normalization': [m.value for m in NormalizationMethod],
                'alignment': [m.value for m in AlignmentMethod],
                'augmentation': True,
                'quality_assessment': True,
                'multi_scale': True,
                'antispoofing': self.enable_antispoofing
            },
            'face_crop_margin': self.face_crop_margin
        }

    def set_clahe_parameters(
        self,
        clip_limit: Optional[float] = None,
        tile_size: Optional[Tuple[int, int]] = None
    ) -> None:
        """Update CLAHE parameters dynamically."""
        if clip_limit is not None:
            if not 0.5 <= clip_limit <= 10.0:
                raise ValueError(
                    f"clip_limit must be between 0.5-10.0, got {clip_limit}"
                )
            self.clahe_clip_limit = clip_limit
        
        if tile_size is not None:
            if len(tile_size) != 2 or tile_size[0] <= 0 or tile_size[1] <= 0:
                raise ValueError(
                    f"tile_size must be tuple of two positive integers, got {tile_size}"
                )
            self.clahe_tile_size = tile_size
        
        if self.use_clahe:
            self._clahe = self._create_clahe()
        
        # Update config
        self._config = self._get_config_dict()

    def __repr__(self) -> str:
        """Developer-friendly string representation."""
        features = []
        if self.enable_photometric_norm:
            features.append("photometric")
        elif self.use_clahe:
            features.append("clahe")
        if self.enable_landmark_alignment:
            features.append("landmark_align")
        elif self.enable_alignment:
            features.append("eye_align")
        if self.enable_quality_rejection:
            features.append("quality_gate")
        if self.enable_antispoofing:
            features.append("antispoofing")
        
        return (
            f"EnhancedFacePreprocessor(size={self.target_size}, "
            f"features=[{', '.join(features)}])"
        )

    def __str__(self) -> str:
        """User-friendly string representation."""
        features = []
        if self.enable_photometric_norm:
            features.append("Photometric Norm")
        elif self.use_clahe:
            features.append("CLAHE")
        if self.enable_landmark_alignment:
            features.append("Landmark Alignment")
        elif self.enable_alignment:
            features.append("Eye Alignment")
        if self.enable_quality_rejection:
            features.append("Quality Gating")
        
        features_str = ", ".join(features) if features else "Basic"
        return f"Enhanced Preprocessor ({features_str}, {self.target_size})"


# ============================================================================
# FACTORY FUNCTIONS FOR COMMON CONFIGURATIONS
# ============================================================================

def create_production_pipeline(
    target_size: Tuple[int, int] = (112, 112),
    enable_antispoofing: bool = True
) -> EnhancedFacePreprocessor:
    """
    Create a production-grade preprocessor with all enhancements.
    
    Features:
    - Facial landmark alignment (MediaPipe)
    - Photometric normalization (Tan-Triggs)
    - Quality rejection
    - Pose filtering
    - Optional anti-spoofing
    
    Args:
        target_size: Output image size (width, height)
        enable_antispoofing: Enable basic anti-spoofing checks
        
    Returns:
        Configured EnhancedFacePreprocessor
    """
    return EnhancedFacePreprocessor(
        target_size=target_size,
        use_clahe=True,
        enable_alignment=True,
        enable_landmark_alignment=True,
        enable_photometric_norm=True,
        enable_quality_rejection=True,
        enable_pose_filtering=True,
        enable_antispoofing=enable_antispoofing,
        quality_rejection_threshold=0.30,
        face_crop_margin=0.2
    )


def create_training_pipeline(
    use_alignment: bool = True,
    use_augmentation: bool = False,
    num_augmentations: int = 5,
    target_size: Tuple[int, int] = (112, 112)
) -> EnhancedFacePreprocessor:
    """
    Create a preprocessor optimized for training data preparation.
    
    Args:
        use_alignment: Enable landmark-based alignment
        use_augmentation: Enable data augmentation
        num_augmentations: Number of augmented samples per image
        target_size: Output image size
        
    Returns:
        Configured EnhancedFacePreprocessor
    """
    preprocessor = EnhancedFacePreprocessor(
        target_size=target_size,
        use_clahe=True,
        clahe_clip_limit=2.0,
        enable_alignment=use_alignment,
        enable_landmark_alignment=True,
        enable_photometric_norm=True,
        enable_quality_rejection=True,
        enable_pose_filtering=True,
        quality_rejection_threshold=0.25,  # More lenient for training
        face_crop_margin=0.2
    )
    
    preprocessor.use_augmentation = use_augmentation
    preprocessor.num_augmentations = num_augmentations
    
    return preprocessor


def create_recognition_pipeline(
    target_size: Tuple[int, int] = (112, 112),
    use_landmark_alignment: bool = True
) -> EnhancedFacePreprocessor:
    """
    Create a preprocessor for real-time recognition/inference.
    
    Args:
        target_size: Output image size
        use_landmark_alignment: Use landmark alignment (more accurate but slower)
        
    Returns:
        Configured EnhancedFacePreprocessor
    """
    return EnhancedFacePreprocessor(
        target_size=target_size,
        use_clahe=True,
        clahe_clip_limit=2.0,
        enable_alignment=True,
        enable_landmark_alignment=use_landmark_alignment,
        enable_photometric_norm=True,
        enable_quality_rejection=True,
        enable_pose_filtering=True,
        quality_rejection_threshold=0.30,
        face_crop_margin=0.15
    )


def create_lightweight_pipeline(
    target_size: Tuple[int, int] = (100, 100)
) -> EnhancedFacePreprocessor:
    """
    Create a lightweight preprocessor for resource-constrained environments.
    
    Minimal processing for speed:
    - No landmark alignment (uses eye-based)
    - Standard CLAHE (no photometric)
    - No quality rejection
    - No pose filtering
    
    Args:
        target_size: Output image size
        
    Returns:
        Configured EnhancedFacePreprocessor
    """
    return EnhancedFacePreprocessor(
        target_size=target_size,
        use_clahe=True,
        clahe_clip_limit=2.0,
        enable_alignment=True,
        enable_landmark_alignment=False,  # Faster
        enable_photometric_norm=False,  # Faster
        enable_quality_rejection=False,  # No rejection
        enable_pose_filtering=False,  # No filtering
        enable_antispoofing=False
    )


def create_high_security_pipeline(
    target_size: Tuple[int, int] = (112, 112)
) -> EnhancedFacePreprocessor:
    """
    Create a high-security preprocessor with strict requirements.
    
    Features:
    - Strict quality requirements
    - Strict pose requirements
    - Anti-spoofing enabled
    - Best preprocessing methods
    
    Args:
        target_size: Output image size
        
    Returns:
        Configured EnhancedFacePreprocessor
    """
    return EnhancedFacePreprocessor(
        target_size=target_size,
        use_clahe=True,
        enable_alignment=True,
        enable_landmark_alignment=True,
        enable_photometric_norm=True,
        enable_quality_rejection=True,
        enable_pose_filtering=True,
        enable_antispoofing=True,
        quality_rejection_threshold=0.50,  # Strict
        face_crop_margin=0.2
    )


def compare_preprocessing_methods(
    face_image: np.ndarray,
    methods: Optional[List[str]] = None
) -> Dict[str, PreprocessingResult]:
    """
    Compare different preprocessing methods on the same image.
    
    Args:
        face_image: Input face image
        methods: List of method names to compare
                Default: ['basic', 'clahe', 'photometric', 'full']
        
    Returns:
        Dictionary mapping method names to preprocessing results
    """
    if methods is None:
        methods = ['basic', 'clahe', 'photometric', 'full']
    
    results = {}
    
    for method in methods:
        if method == 'basic':
            preprocessor = EnhancedFacePreprocessor(
                use_clahe=False,
                enable_alignment=False,
                enable_photometric_norm=False
            )
            
        elif method == 'clahe':
            preprocessor = EnhancedFacePreprocessor(
                use_clahe=True,
                enable_alignment=False,
                enable_photometric_norm=False
            )
            
        elif method == 'photometric':
            preprocessor = EnhancedFacePreprocessor(
                use_clahe=False,
                enable_alignment=False,
                enable_photometric_norm=True
            )
            
        elif method == 'full':
            preprocessor = create_production_pipeline()
        
        else:
            continue
        
        try:
            result = preprocessor.preprocess_with_metadata(face_image)
            results[method] = result
        except Exception as e:
            warnings.warn(f"Method '{method}' failed: {e}", RuntimeWarning)
    
    return results


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_training_inference_match(
    training_preprocessor: EnhancedFacePreprocessor,
    inference_preprocessor: EnhancedFacePreprocessor
) -> bool:
    """
    Validate that training and inference preprocessors match.
    
    Args:
        training_preprocessor: Preprocessor used for training
        inference_preprocessor: Preprocessor used for inference
        
    Returns:
        True if consistent, False otherwise (prints warnings)
    """
    training_config = training_preprocessor.get_preprocessor_info()
    inference_config = inference_preprocessor.get_preprocessor_info()
    
    warnings_list = training_preprocessor.validate_pipeline_consistency(
        inference_config
    )
    
    if warnings_list:
        print("âš ï¸  PIPELINE MISMATCH DETECTED:")
        for warning in warnings_list:
            print(f"  - {warning}")
        return False
    
    print("âœ“ Training and inference pipelines match")
    return True