"""Face detection module using OpenCV Haar Cascades.

Implements the FaceDetector class with comprehensive face detection capabilities,
including quality validation, multi-scale detection, ROI-based detection, simple
tracking, and visualization utilities.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


class DetectionQuality(Enum):
    """Face detection quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


@dataclass
class BoundingBox:
    """Represents a bounding box with utility methods."""
    x: int
    y: int
    width: int
    height: int
    
    @property
    def x1(self) -> int:
        """Left x coordinate."""
        return self.x
    
    @property
    def y1(self) -> int:
        """Top y coordinate."""
        return self.y
    
    @property
    def x2(self) -> int:
        """Right x coordinate."""
        return self.x + self.width
    
    @property
    def y2(self) -> int:
        """Bottom y coordinate."""
        return self.y + self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        """Center point of the bounding box."""
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def area(self) -> int:
        """Area of the bounding box."""
        return self.width * self.height
    
    @property
    def aspect_ratio(self) -> float:
        """Aspect ratio (width / height)."""
        return self.width / self.height if self.height > 0 else 0.0
    
    def as_tuple(self) -> Tuple[int, int, int, int]:
        """Return as (x, y, w, h) tuple."""
        return (self.x, self.y, self.width, self.height)
    
    def as_xyxy(self) -> Tuple[int, int, int, int]:
        """Return as (x1, y1, x2, y2) tuple."""
        return (self.x1, self.y1, self.x2, self.y2)
    
    def iou(self, other: BoundingBox) -> float:
        """Calculate Intersection over Union with another box."""
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = self.area + other.area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def distance_to(self, other: BoundingBox) -> float:
        """Calculate Euclidean distance between centers."""
        cx1, cy1 = self.center
        cx2, cy2 = other.center
        return float(np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2))
    
    def __str__(self) -> str:
        """Human-readable representation."""
        return f"BBox(x={self.x}, y={self.y}, w={self.width}, h={self.height})"


@dataclass
class FaceDetection:
    """Container for a single face detection with metadata."""
    bbox: BoundingBox
    confidence: float = 1.0
    quality_score: float = 0.0
    
    def __str__(self) -> str:
        """Human-readable representation."""
        return f"Face at {self.bbox} (conf: {self.confidence:.2f}, quality: {self.quality_score:.2f})"


@dataclass
class ValidationResult:
    """Container for face validation results."""
    is_valid: bool
    quality_score: float
    quality_level: DetectionQuality
    blur_score: float
    brightness_score: float
    contrast_score: float
    recommendations: List[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        """Human-readable representation."""
        status = "âœ“ Valid" if self.is_valid else "âœ— Invalid"
        return (
            f"{status} ({self.quality_level.value}): "
            f"score={self.quality_score:.2f}, blur={self.blur_score:.1f}, "
            f"brightness={self.brightness_score:.1f}, contrast={self.contrast_score:.1f}"
        )


@dataclass
class TrackingResult:
    """Container for face tracking results."""
    previous_faces: List[BoundingBox]
    current_faces: List[BoundingBox]
    assignments: Dict[int, BoundingBox]
    unmatched_previous: List[int]
    unmatched_current: List[int]
    
    @property
    def tracked_count(self) -> int:
        """Number of successfully tracked faces."""
        return len(self.assignments)
    
    @property
    def lost_count(self) -> int:
        """Number of faces lost from previous frame."""
        return len(self.unmatched_previous)
    
    @property
    def new_count(self) -> int:
        """Number of new faces in current frame."""
        return len(self.unmatched_current)


class FaceDetector:
    """
    Comprehensive face detection system using Haar Cascade classifiers.
    
    This class provides robust face detection with features including:
    - Multi-scale detection with adaptive parameters
    - Quality validation (blur, brightness, contrast)
    - False positive filtering
    - ROI-based detection for performance optimization
    - Simple frame-to-frame tracking
    - Visualization utilities
    
    Example:
        >>> detector = FaceDetector(use_clahe=True, min_neighbors=5)
        >>> faces = detector.detect_faces(image)
        >>> for face in faces:
        ...     print(f"Face detected: {face}")
        
        >>> # With quality validation
        >>> result = detector.validate_face(face_region)
        >>> if result.is_valid:
        ...     print(f"Quality: {result.quality_level.value}")
    """
    
    # Class constants
    DEFAULT_SCALE_FACTOR = 1.1
    DEFAULT_MIN_NEIGHBORS = 5
    DEFAULT_MIN_SIZE = (30, 30)
    
    # Quality thresholds - OPTIMIZED for better face detection
    MIN_FACE_SIZE = 30  # Increased for better quality
    ASPECT_RATIO_MIN = 0.6  # More restrictive for face-like shapes
    ASPECT_RATIO_MAX = 1.8  # More restrictive for face-like shapes
    MIN_BRIGHTNESS = 20  # Increased minimum brightness
    MAX_BRIGHTNESS = 235  # Decreased maximum brightness
    MIN_CONTRAST = 20  # Increased minimum contrast
    MIN_EDGE_DENSITY = 0.03  # Increased edge density requirement
    MIN_FACE_AREA_RATIO = 0.002  # Increased minimum area
    MAX_FACE_AREA_RATIO = 0.25  # Decreased maximum area
    
    # Blur thresholds
    BLUR_THRESHOLD_SHARP = 100.0
    
    # Brightness thresholds
    BRIGHTNESS_MIN = 50.0
    BRIGHTNESS_MAX = 200.0
    
    # Contrast threshold
    CONTRAST_THRESHOLD = 50.0
    
    # Quality score thresholds
    QUALITY_EXCELLENT = 0.9
    QUALITY_GOOD = 0.7
    QUALITY_FAIR = 0.5

    def __init__(
        self,
        cascade_path: str = "haarcascade_frontalface_default.xml",
        scale_factor: float = DEFAULT_SCALE_FACTOR,
        min_neighbors: int = DEFAULT_MIN_NEIGHBORS,
        min_size: Tuple[int, int] = DEFAULT_MIN_SIZE,
        max_size: Tuple[int, int] = (),
        flags: int = 0,
        use_clahe: bool = False,
    ) -> None:
        """
        Initialize the Face Detector.
        
        Args:
            cascade_path: Path to Haar cascade XML file or filename
            scale_factor: Image scale reduction between scales (1.05-1.4, default 1.1)
            min_neighbors: Minimum neighbors required for detection (3-6, default 5)
            min_size: Minimum face size as (width, height) tuple
            max_size: Maximum face size as (width, height) tuple (empty for no limit)
            flags: Operation flags for detection (usually 0 or cv2.CASCADE_SCALE_IMAGE)
            use_clahe: Whether to use CLAHE preprocessing for better detection
            
        Raises:
            ValueError: If parameters are invalid
            IOError: If cascade file cannot be loaded
        """
        self._validate_parameters(scale_factor, min_neighbors, min_size)
        
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.max_size = max_size
        self.flags = flags
        self.use_clahe = use_clahe

        # Initialize CLAHE if requested
        self._clahe = (
            cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            if use_clahe else None
        )
        
        # Load cascade classifier
        self.cascade_path = self._resolve_cascade_path(cascade_path)
        self._cascade = cv2.CascadeClassifier(str(self.cascade_path))
        
        if self._cascade.empty():
            raise IOError(f"Failed to load Haar cascade from: {self.cascade_path}")

    @staticmethod
    def _validate_parameters(
        scale_factor: float,
        min_neighbors: int,
        min_size: Tuple[int, int]
    ) -> None:
        """Validate initialization parameters."""
        if not 1.01 <= scale_factor <= 2.0:
            raise ValueError(
                f"scale_factor must be between 1.01-2.0, got {scale_factor}"
            )
        
        if not 0 <= min_neighbors <= 20:
            raise ValueError(
                f"min_neighbors must be between 0-20, got {min_neighbors}"
            )
        
        if len(min_size) != 2 or min_size[0] <= 0 or min_size[1] <= 0:
            raise ValueError(
                f"min_size must be tuple of two positive integers, got {min_size}"
            )

    @staticmethod
    def _resolve_cascade_path(cascade_path: str) -> Path:
        """Resolve cascade path, using OpenCV data directory if needed."""
        path = Path(cascade_path)
        
        # If path exists, use it directly
        if path.exists():
            return path
        
        # If it's just a filename, look in OpenCV's data directory
        if "/" not in cascade_path and "\\" not in cascade_path:
            opencv_path = Path(cv2.data.haarcascades) / cascade_path
            if opencv_path.exists():
                return opencv_path
        
        # Return original path (will fail in cascade loading)
        return path

    def detect_faces(
        self,
        image: np.ndarray,
        return_gray: bool = False,
        return_confidence: bool = False,
        scale_factor: Optional[float] = None,
        min_neighbors: Optional[int] = None,
        enable_filtering: bool = True,
    ) -> Union[List[Tuple[int, int, int, int]], Tuple]:
        """
        Detect faces in the provided image using Haar Cascade classifier.
        
        Args:
            image: Input image (BGR or grayscale)
            return_gray: Whether to return the grayscale image
            return_confidence: Whether to return confidence scores
            scale_factor: Override default scale factor
            min_neighbors: Override default min neighbors
            enable_filtering: Whether to apply false positive filtering
            
        Returns:
            If return_gray=False and return_confidence=False:
                List of face rectangles as (x, y, w, h) tuples
            Otherwise:
                Tuple containing requested components (faces, [gray], [confidences])
            
        Raises:
            ValueError: If image is None or invalid
        """
        if image is None or image.size == 0:
            raise ValueError("Image must not be None or empty")
        
        # Preprocess image
        gray = self._preprocess_image(image)
        
        # Detect faces
        sf = scale_factor if scale_factor is not None else self.scale_factor
        mn = min_neighbors if min_neighbors is not None else self.min_neighbors

        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=sf,
            minNeighbors=mn,
            minSize=self.min_size,
            maxSize=None if not self.max_size else self.max_size,
            flags=cv2.CASCADE_SCALE_IMAGE if self.flags == 0 else self.flags,
        )
        
        # Convert to list of tuples
        faces_list = [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]
        
        # Apply filtering if enabled
        if enable_filtering:
            faces_list = self._filter_false_positives(faces_list, gray)
        
        # Build return value
        return self._build_detection_output(
            faces_list, gray, return_gray, return_confidence
        )

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for face detection."""
        # Convert to grayscale if needed
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if len(image.shape) == 3
            else image.copy()
        )
        
        # Apply CLAHE and denoising if enabled
        if self._clahe is not None:
            gray = self._clahe.apply(gray)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
        else:
            # Standard histogram equalization
            gray = cv2.equalizeHist(gray)
        
        return gray

    def _filter_false_positives(
        self,
        faces: List[Tuple[int, int, int, int]],
        gray: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """
        Filter out false positive detections using multiple quality checks.
        
        Checks include:
        - Minimum face size
        - Aspect ratio validation
        - Brightness range check
        - Minimum contrast requirement
        - Edge density validation
        - Face area relative to image size
        """
        filtered_faces = []
        img_height, img_width = gray.shape[:2]
        img_area = img_width * img_height
        
        for x, y, w, h in faces:
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Check 1: Minimum size
            if face_roi.size == 0 or w < self.MIN_FACE_SIZE or h < self.MIN_FACE_SIZE:
                continue
            
            # Check 2: Aspect ratio (faces are roughly square to rectangular)
            aspect_ratio = w / h
            if aspect_ratio < self.ASPECT_RATIO_MIN or aspect_ratio > self.ASPECT_RATIO_MAX:
                continue
            
            # Check 3: Brightness range
            mean_brightness = np.mean(face_roi)
            if mean_brightness < self.MIN_BRIGHTNESS or mean_brightness > self.MAX_BRIGHTNESS:
                continue
            
            # Check 4: Minimum contrast
            contrast = np.std(face_roi)
            if contrast < self.MIN_CONTRAST:
                continue
            
            # Check 5: Edge density (faces have facial features = edges)
            edges = cv2.Canny(face_roi, 50, 150)
            edge_density = np.sum(edges > 0) / (w * h)
            if edge_density < self.MIN_EDGE_DENSITY:
                continue
            
            # Check 6: Face area relative to image
            face_area = w * h
            area_ratio = face_area / img_area
            if area_ratio < self.MIN_FACE_AREA_RATIO or area_ratio > self.MAX_FACE_AREA_RATIO:
                continue
            
            filtered_faces.append((x, y, w, h))

        return filtered_faces

    def _build_detection_output(
        self,
        faces: List[Tuple[int, int, int, int]],
        gray: np.ndarray,
        return_gray: bool,
        return_confidence: bool
    ) -> Union[List[Tuple[int, int, int, int]], Tuple]:
        """Build the output tuple based on requested components."""
        if not return_gray and not return_confidence:
            return faces

        outputs = [faces]

        if return_gray:
            outputs.append(gray)

        if return_confidence:
            confidences = self._calculate_confidences(faces, gray)
            outputs.append(confidences)
        
        return tuple(outputs)

    @staticmethod
    def _calculate_confidences(
        faces: List[Tuple[int, int, int, int]],
        gray: np.ndarray
    ) -> List[float]:
        """
        Calculate confidence scores for detected faces.
        
        Note: Haar cascades don't provide confidence, so we use contrast
        as a proxy for face quality/confidence.
        """
        confidences = []
        
        for x, y, w, h in faces:
            roi = gray[y:y+h, x:x+w]
            
            if roi.size == 0:
                confidences.append(0.0)
                continue
            
            # Use contrast as confidence proxy (normalized to 0-1)
            contrast = float(np.std(roi))
            conf = max(0.0, min(1.0, contrast / 64.0))
            confidences.append(conf)

        return confidences

    def detect_faces_multiscale(
        self,
        image: np.ndarray,
        scale_factors: List[float] = None,
        min_neighbors_list: List[int] = None,
        merge_overlapping: bool = True,
        iou_threshold: float = 0.3,
    ) -> List[FaceDetection]:
        """
        Perform multi-scale face detection with different parameters.
        
        Combines results from multiple detection passes with varying parameters
        to improve detection robustness across different face sizes and qualities.
        
        Args:
            image: Input image
            scale_factors: List of scale factors to try (default: [1.05, 1.1, 1.2])
            min_neighbors_list: List of min_neighbors to try (default: [3, 5, 7])
            merge_overlapping: Whether to merge overlapping detections via NMS
            iou_threshold: IoU threshold for merging (0.0-1.0)
            
        Returns:
            List of FaceDetection objects with bounding boxes and confidences
        """
        if scale_factors is None:
            scale_factors = [1.05, 1.1, 1.2]
        if min_neighbors_list is None:
            min_neighbors_list = [3, 5, 7]
        
        detections: List[FaceDetection] = []
        gray = self._preprocess_image(image)
        
        # Run detection with all parameter combinations
        for sf in scale_factors:
            for mn in min_neighbors_list:
                faces = self.detect_faces(
                    image,
                    scale_factor=sf,
                    min_neighbors=mn,
                    enable_filtering=True
                )
                
                for x, y, w, h in faces:
                    bbox = BoundingBox(x, y, w, h)
                    
                    # Calculate confidence from face region
                    roi = gray[y:y+h, x:x+w]
                    confidence = (
                        max(0.0, min(1.0, float(np.std(roi)) / 64.0))
                        if roi.size > 0 else 0.0
                    )
                    
                    detections.append(FaceDetection(bbox, confidence))
        
        # Merge overlapping detections if requested
        if merge_overlapping and detections:
            detections = self._merge_detections(detections, iou_threshold)
        
            return detections

    def _merge_detections(
        self,
        detections: List[FaceDetection],
        iou_threshold: float
    ) -> List[FaceDetection]:
        """Merge overlapping detections using Non-Maximum Suppression."""
        if not detections:
            return []
        
        # Prepare arrays for NMS
        boxes = np.array([det.bbox.as_xyxy() for det in detections], dtype=np.float32)
        scores = np.array([det.confidence for det in detections], dtype=np.float32)
        
        # Apply NMS
        keep_indices = self._non_maximum_suppression(boxes, scores, iou_threshold)
        
        # Return kept detections
        return [detections[i] for i in keep_indices]

    @staticmethod
    def _non_maximum_suppression(
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float
    ) -> List[int]:
        """
        Non-Maximum Suppression algorithm.
        
        Args:
            boxes: Array of boxes in (x1, y1, x2, y2) format
            scores: Array of confidence scores
            iou_threshold: IoU threshold for suppression
            
        Returns:
            List of indices to keep
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        
        keep: List[int] = []
        
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            
            # Calculate IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            intersection = w * h
            
            iou = intersection / (areas[i] + areas[order[1:]] - intersection + 1e-6)
            
            # Keep boxes with IoU below threshold
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep

    def validate_face(
        self,
        face_region: np.ndarray,
        min_quality_threshold: float = QUALITY_GOOD,
        check_blur: bool = True,
        check_brightness: bool = True,
        check_contrast: bool = True,
    ) -> ValidationResult:
        """
        Validate the quality of a detected face region using multiple metrics.
        
        Args:
            face_region: Face image region to validate
            min_quality_threshold: Minimum quality score required (0.0-1.0)
            check_blur: Whether to check for blur
            check_brightness: Whether to check brightness levels
            check_contrast: Whether to check contrast
            
        Returns:
            ValidationResult object with detailed quality assessment
        """
        if face_region is None or face_region.size == 0:
            return ValidationResult(
                is_valid=False,
                quality_score=0.0,
                quality_level=DetectionQuality.POOR,
                blur_score=0.0,
                brightness_score=0.0,
                contrast_score=0.0,
                recommendations=["Invalid or empty face region"]
            )
        
        # Convert to grayscale
        gray = (
            cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            if len(face_region.shape) == 3
            else face_region.copy()
        )

        recommendations: List[str] = []

        # Blur assessment
        blur_score, is_sharp = self._assess_blur(gray, check_blur)
        if check_blur and not is_sharp:
            recommendations.append("Image is blurry - ensure camera is focused")

        # Brightness assessment
        brightness_score, brightness_ok = self._assess_brightness(gray, check_brightness)
        if check_brightness and not brightness_ok:
            if brightness_score < self.BRIGHTNESS_MIN:
                recommendations.append("Image is too dark - improve lighting")
            else:
                recommendations.append("Image is too bright - reduce lighting")

        # Contrast assessment
        contrast_score, contrast_ok = self._assess_contrast(gray, check_contrast)
        if check_contrast and not contrast_ok:
            recommendations.append("Low contrast - adjust lighting or camera settings")

        # Calculate overall quality score
        factors = [is_sharp, brightness_ok, contrast_ok]
        quality_score = sum(1.0 for f in factors if f) / len(factors)

        # Determine quality level
        quality_level = self._determine_quality_level(quality_score)
        
        return ValidationResult(
            is_valid=quality_score >= min_quality_threshold,
            quality_score=quality_score,
            quality_level=quality_level,
            blur_score=blur_score,
            brightness_score=brightness_score,
            contrast_score=contrast_score,
            recommendations=recommendations
        )

    def _assess_blur(
        self,
        gray: np.ndarray,
        check_blur: bool
    ) -> Tuple[float, bool]:
        """Assess blur using Laplacian variance."""
        if not check_blur:
            return 999.0, True
        
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        is_sharp = blur_score > self.BLUR_THRESHOLD_SHARP
        
        return blur_score, is_sharp

    def _assess_brightness(
        self,
        gray: np.ndarray,
        check_brightness: bool
    ) -> Tuple[float, bool]:
        """Assess brightness levels."""
        if not check_brightness:
            return 128.0, True
        
        brightness_score = float(np.mean(gray))
        brightness_ok = self.BRIGHTNESS_MIN <= brightness_score <= self.BRIGHTNESS_MAX
        
        return brightness_score, brightness_ok

    def _assess_contrast(
        self,
        gray: np.ndarray,
        check_contrast: bool
    ) -> Tuple[float, bool]:
        """Assess contrast levels."""
        if not check_contrast:
            return 64.0, True
        
        contrast_score = float(np.std(gray))
        contrast_ok = contrast_score > self.CONTRAST_THRESHOLD
        
        return contrast_score, contrast_ok

    @staticmethod
    def _determine_quality_level(quality_score: float) -> DetectionQuality:
        """Determine quality level from score."""
        if quality_score >= FaceDetector.QUALITY_EXCELLENT:
            return DetectionQuality.EXCELLENT
        elif quality_score >= FaceDetector.QUALITY_GOOD:
            return DetectionQuality.GOOD
        elif quality_score >= FaceDetector.QUALITY_FAIR:
            return DetectionQuality.FAIR
        else:
            return DetectionQuality.POOR

    def draw_faces(
        self,
        image: np.ndarray,
        faces: Union[List[Tuple[int, int, int, int]], List[FaceDetection]],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        show_confidence: bool = False,
        confidences: Optional[List[float]] = None,
        font_scale: float = 0.5,
        show_index: bool = False,
    ) -> np.ndarray:
        """
        Draw rectangles around detected faces with optional annotations.
        
        Args:
            image: Image to draw on
            faces: List of face rectangles or FaceDetection objects
            color: Rectangle color as BGR tuple
            thickness: Rectangle line thickness
            show_confidence: Whether to show confidence scores
            confidences: Optional list of confidence scores (if not using FaceDetection)
            font_scale: Font scale for text
            show_index: Whether to show face index numbers
            
        Returns:
            Image with drawn faces
        """
        output = image.copy()
        
        for idx, face in enumerate(faces):
            # Extract bbox and confidence
            if isinstance(face, FaceDetection):
                bbox = face.bbox.as_tuple()
                conf = face.confidence
            else:
                bbox = face
                conf = confidences[idx] if confidences and idx < len(confidences) else None
            
            x, y, w, h = bbox
            
            # Draw rectangle
            cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)
            
            # Prepare text annotations
            texts = []
            if show_index:
                texts.append(f"#{idx}")
            if show_confidence and conf is not None:
                texts.append(f"{conf:.2f}")
            
            # Draw text if any
            if texts:
                text = " ".join(texts)
                text_y = max(y - 5, 15)
                cv2.putText(
                    output,
                    text,
                    (x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    1,
                    cv2.LINE_AA,
                )
        
        return output

    def detect_faces_roi(
        self,
        image: np.ndarray,
        roi_regions: List[Tuple[int, int, int, int]],
        adaptive_params: bool = True
    ) -> Dict[int, List[BoundingBox]]:
        """
        Detect faces within specific regions of interest for improved performance.
        
        Useful for tracking scenarios where face locations are approximately known,
        or for processing specific areas of large images.
        
        Args:
            image: Input image
            roi_regions: List of ROI regions as (x, y, w, h) tuples
            adaptive_params: Whether to adapt detection parameters based on ROI size
            
        Returns:
            Dictionary mapping ROI index to list of detected face BoundingBoxes
        """
        if image is None or image.size == 0:
            return {}
        
        results: Dict[int, List[BoundingBox]] = {}

        for idx, (rx, ry, rw, rh) in enumerate(roi_regions):
            # Extract ROI
            sub_img = image[ry:ry+rh, rx:rx+rw]
            
            if sub_img.size == 0:
                results[idx] = []
                continue

            # Determine detection parameters
            if adaptive_params:
                sf, mn = self._adapt_params_for_roi(rw, rh)
            else:
                sf = self.scale_factor
                mn = self.min_neighbors
            
            # Detect faces in ROI
            faces = self.detect_faces(
                sub_img,
                scale_factor=sf,
                min_neighbors=mn,
                enable_filtering=True
            )
            
            # Translate coordinates back to full image space
            translated = [
                BoundingBox(x + rx, y + ry, w, h)
                for x, y, w, h in faces
            ]
            results[idx] = translated

        return results

    def _adapt_params_for_roi(
        self,
        roi_width: int,
        roi_height: int
    ) -> Tuple[float, int]:
        """
        Adapt detection parameters based on ROI size.
        
        Larger ROIs can use coarser scales, smaller ROIs need finer scales
        for better detection.
        """
        roi_area = roi_width * roi_height
        
        if roi_area > 400 * 400:  # Large ROI
            scale_factor = 1.15
            min_neighbors = max(4, self.min_neighbors)
        elif roi_area < 120 * 120:  # Small ROI
            scale_factor = 1.05
            min_neighbors = max(3, self.min_neighbors - 1)
        else:  # Medium ROI
            scale_factor = self.scale_factor
            min_neighbors = self.min_neighbors
        
        return scale_factor, min_neighbors

    def track_faces(
        self,
        previous_faces: List[Union[Tuple[int, int, int, int], BoundingBox]],
        current_image: np.ndarray,
        max_displacement: int = 50,
        iou_weight: float = 0.3,
    ) -> TrackingResult:
        """
        Track faces across consecutive frames using hybrid distance and IoU matching.
        
        This method associates detected faces in the current frame with faces from
        the previous frame using a combination of:
        - Center point distance (Euclidean)
        - Intersection over Union (IoU) overlap
        
        Args:
            previous_faces: List of face bounding boxes from previous frame
            current_image: Current frame image
            max_displacement: Maximum allowed center displacement for matching
            iou_weight: Weight for IoU in matching (0.0-1.0, default 0.3)
            
        Returns:
            TrackingResult object with detailed tracking information
        """
        # Convert previous faces to BoundingBox objects
        prev_boxes = [
            face if isinstance(face, BoundingBox)
            else BoundingBox(*face)
            for face in previous_faces
        ]
        
        # Detect faces in current frame
        current_faces_raw = self.detect_faces(current_image)
        current_boxes = [BoundingBox(*face) for face in current_faces_raw]
        
        # Perform matching
        assignments, unmatched_prev, unmatched_curr = self._match_faces(
            prev_boxes,
            current_boxes,
            max_displacement,
            iou_weight
        )
        
        return TrackingResult(
            previous_faces=prev_boxes,
            current_faces=current_boxes,
            assignments=assignments,
            unmatched_previous=unmatched_prev,
            unmatched_current=unmatched_curr
        )

    def _match_faces(
        self,
        previous_boxes: List[BoundingBox],
        current_boxes: List[BoundingBox],
        max_displacement: int,
        iou_weight: float
    ) -> Tuple[Dict[int, BoundingBox], List[int], List[int]]:
        """
        Match faces between frames using hybrid distance+IoU metric.
        
        Returns:
            Tuple of (assignments, unmatched_previous, unmatched_current)
        """
        assigned: Dict[int, BoundingBox] = {}
        used_current: set[int] = set()

        # Greedy matching: for each previous face, find best current match
        for prev_idx, prev_box in enumerate(previous_boxes):
            best_curr_idx = -1
            best_score = float('inf')
            
            for curr_idx, curr_box in enumerate(current_boxes):
                if curr_idx in used_current:
                    continue
                
                # Calculate hybrid matching score
                score = self._calculate_matching_score(
                    prev_box,
                    curr_box,
                    iou_weight
                )
                
                if score < best_score:
                    best_score = score
                    best_curr_idx = curr_idx
            
            # Assign if within displacement threshold
            if best_curr_idx != -1:
                distance = previous_boxes[prev_idx].distance_to(
                    current_boxes[best_curr_idx]
                )
                if distance <= max_displacement:
                    assigned[prev_idx] = current_boxes[best_curr_idx]
                    used_current.add(best_curr_idx)
        
        # Find unmatched faces
        unmatched_previous = [
            i for i in range(len(previous_boxes))
            if i not in assigned
        ]
        unmatched_current = [
            i for i in range(len(current_boxes))
            if i not in used_current
        ]
        
        return assigned, unmatched_previous, unmatched_current

    @staticmethod
    def _calculate_matching_score(
        box1: BoundingBox,
        box2: BoundingBox,
        iou_weight: float
    ) -> float:
        """
        Calculate hybrid matching score combining distance and IoU.
        
        Lower score = better match
        """
        # Euclidean distance between centers
        distance = box1.distance_to(box2)
        
        # IoU (higher is better, so we use 1 - IoU)
        iou = box1.iou(box2)
        iou_score = 1.0 - iou
        
        # Weighted combination
        distance_weight = 1.0 - iou_weight
        score = distance_weight * distance + iou_weight * iou_score * 100
        
        return score

    def get_detector_info(self) -> Dict[str, any]:
        """
        Get comprehensive detector configuration and statistics.
        
        Returns:
            Dictionary containing detector information
        """
        return {
            'cascade_path': str(self.cascade_path),
            'parameters': {
                'scale_factor': self.scale_factor,
                'min_neighbors': self.min_neighbors,
                'min_size': self.min_size,
                'max_size': self.max_size if self.max_size else None,
                'use_clahe': self.use_clahe,
            },
            'quality_thresholds': {
                'min_face_size': self.MIN_FACE_SIZE,
                'aspect_ratio_range': (self.ASPECT_RATIO_MIN, self.ASPECT_RATIO_MAX),
                'brightness_range': (self.MIN_BRIGHTNESS, self.MAX_BRIGHTNESS),
                'min_contrast': self.MIN_CONTRAST,
                'blur_threshold': self.BLUR_THRESHOLD_SHARP,
            },
            'cascade_loaded': not self._cascade.empty()
        }

    def set_parameters(
        self,
        scale_factor: Optional[float] = None,
        min_neighbors: Optional[int] = None,
        min_size: Optional[Tuple[int, int]] = None,
        max_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Update detection parameters dynamically.
        
        Args:
            scale_factor: New scale factor (1.01-2.0)
            min_neighbors: New min neighbors (0-20)
            min_size: New minimum size tuple
            max_size: New maximum size tuple
            
        Raises:
            ValueError: If any parameter is invalid
        """
        if scale_factor is not None:
            if not 1.01 <= scale_factor <= 2.0:
                raise ValueError(
                    f"scale_factor must be between 1.01-2.0, got {scale_factor}"
                )
            self.scale_factor = scale_factor
        
        if min_neighbors is not None:
            if not 0 <= min_neighbors <= 20:
                raise ValueError(
                    f"min_neighbors must be between 0-20, got {min_neighbors}"
                )
            self.min_neighbors = min_neighbors
        
        if min_size is not None:
            if len(min_size) != 2 or min_size[0] <= 0 or min_size[1] <= 0:
                raise ValueError(
                    f"min_size must be tuple of two positive integers, got {min_size}"
                )
            self.min_size = min_size
        
        if max_size is not None:
            if max_size and (len(max_size) != 2 or max_size[0] <= 0 or max_size[1] <= 0):
                raise ValueError(
                    f"max_size must be empty tuple or two positive integers, got {max_size}"
                )
            self.max_size = max_size

    def optimize_parameters(
        self,
        test_images: List[np.ndarray],
        ground_truth_counts: Optional[List[int]] = None,
        parameter_grid: Optional[Dict[str, List]] = None,
    ) -> Dict[str, any]:
        """
        Optimize detection parameters using test images.
        
        Tests different parameter combinations and selects the best based on:
        - Detection count consistency
        - Agreement with ground truth (if provided)
        - Average confidence scores
        
        Args:
            test_images: List of test images
            ground_truth_counts: Optional list of expected face counts per image
            parameter_grid: Optional custom parameter grid to search
                          Default: {'scale_factor': [1.05, 1.1, 1.15, 1.2],
                                   'min_neighbors': [3, 4, 5, 6, 7]}
            
        Returns:
            Dictionary with best parameters and optimization results
        """
        if not test_images:
            raise ValueError("test_images cannot be empty")
        
        # Default parameter grid
        if parameter_grid is None:
            parameter_grid = {
                'scale_factor': [1.05, 1.1, 1.15, 1.2],
                'min_neighbors': [3, 4, 5, 6, 7]
            }
        
        best_score = -float('inf')
        best_params = None
        results = []
        
        # Test all parameter combinations
        for sf in parameter_grid.get('scale_factor', [self.scale_factor]):
            for mn in parameter_grid.get('min_neighbors', [self.min_neighbors]):
                # Evaluate this parameter combination
                score, metrics = self._evaluate_parameters(
                    test_images,
                    ground_truth_counts,
                    sf,
                    mn
                )
                
                results.append({
                    'scale_factor': sf,
                    'min_neighbors': mn,
                    'score': score,
                    'metrics': metrics
                })
                
                if score > best_score:
                    best_score = score
                    best_params = {'scale_factor': sf, 'min_neighbors': mn}
        
        # Update detector with best parameters
        if best_params:
            self.scale_factor = best_params['scale_factor']
            self.min_neighbors = best_params['min_neighbors']

        return {
            'best_parameters': best_params,
            'best_score': best_score,
            'all_results': results,
            'updated': best_params is not None
        }

    def _evaluate_parameters(
        self,
        test_images: List[np.ndarray],
        ground_truth_counts: Optional[List[int]],
        scale_factor: float,
        min_neighbors: int
    ) -> Tuple[float, Dict[str, float]]:
        """Evaluate a parameter combination on test images."""
        detection_counts = []
        confidences = []
        
        for img in test_images:
            faces, _, confs = self.detect_faces(
                img,
                return_gray=True,
                return_confidence=True,
                scale_factor=scale_factor,
                min_neighbors=min_neighbors
            )
            detection_counts.append(len(faces))
            confidences.extend(confs)
        
        # Calculate metrics
        avg_detections = np.mean(detection_counts)
        std_detections = np.std(detection_counts)
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Calculate score
        score = avg_confidence * 100  # Base score from confidence
        
        # Penalty for high variance (inconsistent detections)
        score -= std_detections * 10
        
        # Bonus if ground truth provided and matches well
        if ground_truth_counts:
            accuracy = sum(
                1 for det, gt in zip(detection_counts, ground_truth_counts)
                if det == gt
            ) / len(ground_truth_counts)
            score += accuracy * 50
        
        metrics = {
            'avg_detections': avg_detections,
            'std_detections': std_detections,
            'avg_confidence': avg_confidence
        }
        
        if ground_truth_counts:
            metrics['accuracy'] = accuracy
        
        return score, metrics

    def batch_detect(
        self,
        images: List[np.ndarray],
        enable_filtering: bool = True,
        show_progress: bool = False
    ) -> List[List[Tuple[int, int, int, int]]]:
        """
        Detect faces in multiple images efficiently.
        
        Args:
            images: List of images to process
            enable_filtering: Whether to apply false positive filtering
            show_progress: Whether to print progress (useful for large batches)
            
        Returns:
            List of face detection lists (one per image)
        """
        results = []
        
        for idx, img in enumerate(images):
            if show_progress and idx % 10 == 0:
                print(f"Processing image {idx + 1}/{len(images)}...")
            
            faces = self.detect_faces(img, enable_filtering=enable_filtering)
            results.append(faces)
        
        if show_progress:
            total_faces = sum(len(faces) for faces in results)
            print(f"Completed: {len(images)} images, {total_faces} faces detected")
        
        return results

    def __repr__(self) -> str:
        """Developer-friendly string representation."""
        return (
            f"FaceDetector(cascade={Path(self.cascade_path).name}, "
            f"scale={self.scale_factor}, neighbors={self.min_neighbors}, "
            f"clahe={self.use_clahe})"
        )

    def __str__(self) -> str:
        """User-friendly string representation."""
        preprocessing = "CLAHE enabled" if self.use_clahe else "Standard"
        return (
            f"FaceDetector using {Path(self.cascade_path).name} "
            f"({preprocessing}, min_size={self.min_size})"
        )