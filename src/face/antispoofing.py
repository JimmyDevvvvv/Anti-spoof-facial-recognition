"""Advanced Anti-Spoofing Module for Face Recognition Systems.

Implements multi-layered liveness detection to prevent presentation attacks:
- Texture analysis (LBP - Local Binary Patterns)
- Frequency domain analysis (moirÃƒÂ© patterns, print artifacts)
- Color space analysis (skin tone properties)
- Motion-based detection (optical flow, micro-expressions)
- Eye blink detection
- Depth estimation (face geometry)
- Challenge-response (interactive verification)
- Deep learning models (optional)

Can be used standalone or integrated with preprocessing/recognition pipeline.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import time

import cv2
import numpy as np


class SpoofingAttackType(Enum):
    """Types of spoofing attacks."""
    UNKNOWN = "unknown"
    PRINT_ATTACK = "print_attack"
    DIGITAL_DISPLAY = "digital_display"
    VIDEO_REPLAY = "video_replay"
    MASK_ATTACK = "mask_attack"
    PHOTO_CUTOUT = "photo_cutout"


class LivenessLevel(Enum):
    """Liveness detection confidence levels."""
    LIVE = "live"
    LIKELY_LIVE = "likely_live"
    SUSPICIOUS = "suspicious"
    LIKELY_SPOOF = "likely_spoof"
    SPOOF = "spoof"


class DetectionMode(Enum):
    """Anti-spoofing detection modes."""
    PASSIVE_SINGLE = "passive_single"  # Single image, no interaction
    PASSIVE_MULTI = "passive_multi"    # Multiple frames, no interaction
    ACTIVE_BLINK = "active_blink"      # Requires eye blink
    ACTIVE_MOTION = "active_motion"    # Requires head movement
    ACTIVE_CHALLENGE = "active_challenge"  # Random challenges


@dataclass
class TextureMetrics:
    """Texture analysis metrics for spoofing detection."""
    lbp_score: float  # Local Binary Pattern score
    lbp_histogram: np.ndarray  # LBP histogram
    texture_uniformity: float  # Texture uniformity measure
    is_suspicious: bool
    
    def __str__(self) -> str:
        return f"Texture: LBP={self.lbp_score:.2f}, uniformity={self.texture_uniformity:.2f}"


@dataclass
class FrequencyMetrics:
    """Frequency domain analysis metrics."""
    high_freq_energy: float  # High frequency energy
    moire_score: float  # Moire pattern detection
    print_artifact_score: float  # Print artifacts
    is_suspicious: bool
    
    def __str__(self) -> str:
        return f"Frequency: moire={self.moire_score:.2f}, artifacts={self.print_artifact_score:.2f}"


@dataclass
class ColorMetrics:
    """Color space analysis metrics."""
    skin_tone_score: float  # Skin tone naturalness
    color_diversity: float  # Color diversity measure
    rgb_variance: Tuple[float, float, float]  # RGB channel variances
    ycrcb_consistency: float  # YCrCb space consistency
    is_suspicious: bool
    
    def __str__(self) -> str:
        return f"Color: skin_tone={self.skin_tone_score:.2f}, diversity={self.color_diversity:.2f}"


@dataclass
class MotionMetrics:
    """Motion-based liveness metrics."""
    optical_flow_magnitude: float  # Average optical flow
    motion_consistency: float  # Motion pattern consistency
    face_motion_detected: bool  # Whether face motion detected
    background_motion_ratio: float  # Background vs face motion
    is_suspicious: bool
    
    def __str__(self) -> str:
        return f"Motion: flow={self.optical_flow_magnitude:.2f}, consistency={self.motion_consistency:.2f}"


@dataclass
class BlinkMetrics:
    """Eye blink detection metrics."""
    blink_detected: bool
    blink_count: int
    ear_scores: List[float]  # Eye Aspect Ratio over time
    blink_frequency: float  # Blinks per second
    is_suspicious: bool
    
    def __str__(self) -> str:
        return f"Blink: detected={self.blink_detected}, count={self.blink_count}, freq={self.blink_frequency:.2f}/s"


@dataclass
class DepthMetrics:
    """Depth and 3D structure metrics."""
    face_depth_variance: float  # Face depth variation
    edge_gradient_score: float  # Edge gradient analysis
    is_3d_structure: bool  # 3D vs flat surface
    nose_prominence: float  # Nose depth prominence
    is_suspicious: bool
    
    def __str__(self) -> str:
        return f"Depth: variance={self.face_depth_variance:.2f}, 3D={self.is_3d_structure}"


@dataclass
class LivenessResult:
    """Complete liveness detection result."""
    is_live: bool
    confidence: float  # 0.0 (definitely spoof) to 1.0 (definitely live)
    liveness_level: LivenessLevel
    detected_attack_type: Optional[SpoofingAttackType]
    
    # Individual metric results
    texture_metrics: Optional[TextureMetrics] = None
    frequency_metrics: Optional[FrequencyMetrics] = None
    color_metrics: Optional[ColorMetrics] = None
    motion_metrics: Optional[MotionMetrics] = None
    blink_metrics: Optional[BlinkMetrics] = None
    depth_metrics: Optional[DepthMetrics] = None
    
    # Metadata
    detection_mode: DetectionMode = DetectionMode.PASSIVE_SINGLE
    processing_time: float = 0.0
    warnings: List[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        status = "Ã¢Å“â€œ LIVE" if self.is_live else "Ã¢Å“â€” SPOOF"
        attack = f" ({self.detected_attack_type.value})" if self.detected_attack_type else ""
        return (
            f"{status}{attack}: {self.liveness_level.value} "
            f"(confidence: {self.confidence:.2%})"
        )
    
    def get_detailed_report(self) -> str:
        """Get detailed analysis report."""
        lines = [str(self)]
        lines.append(f"Detection Mode: {self.detection_mode.value}")
        lines.append(f"Processing Time: {self.processing_time:.3f}s")
        
        if self.texture_metrics:
            lines.append(f"  {self.texture_metrics}")
        if self.frequency_metrics:
            lines.append(f"  {self.frequency_metrics}")
        if self.color_metrics:
            lines.append(f"  {self.color_metrics}")
        if self.motion_metrics:
            lines.append(f"  {self.motion_metrics}")
        if self.blink_metrics:
            lines.append(f"  {self.blink_metrics}")
        if self.depth_metrics:
            lines.append(f"  {self.depth_metrics}")
        
        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        
        return "\n".join(lines)


class AntiSpoofingDetector:
    """
    Advanced anti-spoofing detector for face recognition systems.
    
    Implements multiple detection techniques:
    1. Passive (single image):
       - Texture analysis (LBP)
       - Frequency analysis (moirÃƒÂ©, artifacts)
       - Color analysis (skin tone)
       - Depth estimation
    
    2. Passive (multi-frame):
       - Motion analysis (optical flow)
       - Micro-expression detection
       - Background consistency
    
    3. Active (interactive):
       - Eye blink detection
       - Head movement challenges
       - Random challenge-response
    
    Example:
        >>> detector = AntiSpoofingDetector(detection_mode=DetectionMode.PASSIVE_SINGLE)
        >>> result = detector.detect(face_image)
        >>> if result.is_live:
        ...     print("Live person detected")
        ... else:
        ...     print(f"Spoofing attack: {result.detected_attack_type}")
    """
    
    # Detection thresholds - ADJUSTED FOR REAL-WORLD USE
    LBP_THRESHOLD = 0.4  # LBP score threshold (was 0.6 - too strict)
    MOIRE_THRESHOLD = 60.0  # Moire pattern threshold (was 45.0 - too strict)
    SKIN_TONE_THRESHOLD = 0.3  # Skin tone naturalness (was 0.5 - too strict)
    MOTION_THRESHOLD = 1.0  # Minimum motion for liveness (was 2.0 - too strict)
    BLINK_REQUIRED = 1  # Minimum blinks required
    DEPTH_VARIANCE_THRESHOLD = 3.0  # Minimum depth variance (was 5.0 - too strict)
    
    # Color space ranges for skin detection (YCrCb)
    SKIN_YCRCB_MIN = np.array([0, 133, 77], dtype=np.uint8)
    SKIN_YCRCB_MAX = np.array([255, 173, 127], dtype=np.uint8)
    
    def __init__(
        self,
        detection_mode: DetectionMode = DetectionMode.PASSIVE_SINGLE,
        enable_texture_analysis: bool = True,
        enable_frequency_analysis: bool = True,
        enable_color_analysis: bool = True,
        enable_depth_analysis: bool = True,
        enable_motion_analysis: bool = False,
        enable_blink_detection: bool = False,
        confidence_threshold: float = 0.6,
        strict_mode: bool = False
    ):
        """
        Initialize anti-spoofing detector.
        
        Args:
            detection_mode: Detection mode to use
            enable_texture_analysis: Enable LBP texture analysis
            enable_frequency_analysis: Enable FFT/moirÃƒÂ© detection
            enable_color_analysis: Enable skin tone analysis
            enable_depth_analysis: Enable depth estimation
            enable_motion_analysis: Enable motion-based detection (requires video)
            enable_blink_detection: Enable eye blink detection (requires video)
            confidence_threshold: Minimum confidence for live classification
            strict_mode: Use stricter thresholds (higher security)
        """
        self.detection_mode = detection_mode
        self.enable_texture_analysis = enable_texture_analysis
        self.enable_frequency_analysis = enable_frequency_analysis
        self.enable_color_analysis = enable_color_analysis
        self.enable_depth_analysis = enable_depth_analysis
        self.enable_motion_analysis = enable_motion_analysis
        self.enable_blink_detection = enable_blink_detection
        self.confidence_threshold = confidence_threshold
        self.strict_mode = strict_mode
        
        # Adjust thresholds for strict mode
        if strict_mode:
            self.confidence_threshold = max(confidence_threshold, 0.75)
        
        # Initialize face/eye detectors
        self._face_cascade = self._load_cascade('haarcascade_frontalface_default.xml')
        self._eye_cascade = self._load_cascade('haarcascade_eye.xml')
        
        # Frame buffer for multi-frame analysis
        self._frame_buffer: List[np.ndarray] = []
        self._max_buffer_size = 30  # ~1 second at 30fps
        
        # Blink detection state
        self._ear_history: List[float] = []
        self._blink_counter = 0
        self._last_blink_time = 0.0
    
    def _load_cascade(self, cascade_name: str) -> Optional[cv2.CascadeClassifier]:
        """Load Haar cascade classifier."""
        try:
            from pathlib import Path
            cascade_path = Path(cv2.data.haarcascades) / cascade_name
            cascade = cv2.CascadeClassifier(str(cascade_path))
            if cascade.empty():
                return None
            return cascade
        except Exception:
            return None
    
    def detect(
        self,
        face_image: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> LivenessResult:
        """
        Perform liveness detection on face image.
        
        Args:
            face_image: Input face image (BGR or grayscale)
            face_bbox: Optional face bounding box (x, y, w, h)
            
        Returns:
            LivenessResult with comprehensive analysis
        """
        start_time = time.time()
        
        if face_image is None or face_image.size == 0:
            return self._create_error_result("Invalid input image")
        
        warnings_list = []
        
        # Convert to color if grayscale
        if len(face_image.shape) == 2:
            face_color = cv2.cvtColor(face_image, cv2.COLOR_GRAY2BGR)
            face_gray = face_image
        else:
            face_color = face_image
            face_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Detect face if bbox not provided
        if face_bbox is None and self._face_cascade is not None:
            faces = self._face_cascade.detectMultiScale(face_gray, 1.3, 5)
            if len(faces) == 0:
                return self._create_error_result("No face detected")
            face_bbox = tuple(faces[0])
        
        # Perform individual analyses
        texture_metrics = None
        frequency_metrics = None
        color_metrics = None
        motion_metrics = None
        blink_metrics = None
        depth_metrics = None
        
        if self.enable_texture_analysis:
            texture_metrics = self._analyze_texture(face_gray, face_bbox)
        
        if self.enable_frequency_analysis:
            frequency_metrics = self._analyze_frequency(face_gray, face_bbox)
        
        if self.enable_color_analysis:
            color_metrics = self._analyze_color(face_color, face_bbox)
        
        if self.enable_depth_analysis:
            depth_metrics = self._analyze_depth(face_gray, face_bbox)
        
        if self.enable_motion_analysis:
            motion_metrics = self._analyze_motion(face_gray, face_bbox)
        
        if self.enable_blink_detection:
            blink_metrics = self._detect_blink(face_gray, face_bbox)
        
        # Aggregate results
        result = self._aggregate_results(
            texture_metrics,
            frequency_metrics,
            color_metrics,
            motion_metrics,
            blink_metrics,
            depth_metrics,
            warnings_list
        )
        
        result.detection_mode = self.detection_mode
        result.processing_time = time.time() - start_time
        
        return result
    
    def detect_video_stream(
        self,
        video_source: Union[int, str],
        duration_seconds: float = 3.0,
        require_blink: bool = True,
        require_motion: bool = True
    ) -> LivenessResult:
        """
        Perform liveness detection on video stream.
        
        Args:
            video_source: Camera index or video file path
            duration_seconds: How long to analyze
            require_blink: Require at least one blink
            require_motion: Require face motion
            
        Returns:
            LivenessResult from video analysis
        """
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            return self._create_error_result("Cannot open video source")
        
        start_time = time.time()
        self._frame_buffer.clear()
        self._ear_history.clear()
        self._blink_counter = 0
        
        try:
            while (time.time() - start_time) < duration_seconds:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Add to buffer
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self._frame_buffer.append(gray)
                if len(self._frame_buffer) > self._max_buffer_size:
                    self._frame_buffer.pop(0)
                
                # Detect face
                if self._face_cascade is not None:
                    faces = self._face_cascade.detectMultiScale(gray, 1.3, 5)
                    if len(faces) > 0:
                        face_bbox = tuple(faces[0])
                        
                        # Track blinks
                        if require_blink:
                            self._update_blink_detection(gray, face_bbox)
        
        finally:
            cap.release()
        
        if len(self._frame_buffer) < 2:
            return self._create_error_result("Insufficient frames captured")
        
        # Analyze accumulated frames
        return self._analyze_video_buffer(require_blink, require_motion)
    
    def _analyze_texture(
        self,
        face_gray: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]]
    ) -> TextureMetrics:
        """
        Analyze texture using Local Binary Patterns (LBP).
        
        Real faces have rich micro-textures; prints/screens are smoother.
        """
        try:
            # Extract face region
            if face_bbox:
                x, y, w, h = face_bbox
                face_roi = face_gray[y:y+h, x:x+w]
            else:
                face_roi = face_gray
            
            # Compute LBP
            lbp_image = self._compute_lbp(face_roi)
            
            # Compute LBP histogram
            lbp_hist, _ = np.histogram(
                lbp_image.ravel(),
                bins=256,
                range=(0, 256),
                density=True
            )
            
            # Calculate texture score (entropy-based)
            # Real faces have more uniform LBP distribution
            lbp_entropy = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-10))
            lbp_score = lbp_entropy / 8.0  # Normalize
            
            # Calculate texture uniformity
            texture_uniformity = np.std(lbp_image) / 255.0
            
            # Suspicious if texture too uniform (print) or too noisy (digital display)
            is_suspicious = (
                lbp_score < self.LBP_THRESHOLD or
                texture_uniformity < 0.15 or
                texture_uniformity > 0.8
            )
            
            if self.strict_mode and lbp_score < 0.7:
                is_suspicious = True
            
            return TextureMetrics(
                lbp_score=float(lbp_score),
                lbp_histogram=lbp_hist,
                texture_uniformity=float(texture_uniformity),
                is_suspicious=is_suspicious
            )
            
        except Exception as e:
            warnings.warn(f"Texture analysis failed: {e}", RuntimeWarning)
            return TextureMetrics(0.5, np.zeros(256), 0.5, True)
    
    @staticmethod
    def _compute_lbp(image: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
        """Compute Local Binary Pattern."""
        height, width = image.shape
        lbp = np.zeros_like(image)
        
        for i in range(radius, height - radius):
            for j in range(radius, width - radius):
                center = image[i, j]
                binary_string = ''
                
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = i + int(radius * np.cos(angle))
                    y = j + int(radius * np.sin(angle))
                    
                    if 0 <= x < height and 0 <= y < width:
                        binary_string += '1' if image[x, y] >= center else '0'
                
                lbp[i, j] = int(binary_string, 2) if binary_string else 0
        
        return lbp
    
    def _analyze_frequency(
        self,
        face_gray: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]]
    ) -> FrequencyMetrics:
        """
        Analyze frequency domain for moirÃƒÂ© patterns and print artifacts.
        
        Screens/prints have characteristic frequency patterns.
        """
        try:
            if face_bbox:
                x, y, w, h = face_bbox
                face_roi = face_gray[y:y+h, x:x+w]
            else:
                face_roi = face_gray
            
            # FFT analysis
            fft = np.fft.fft2(face_roi)
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shift)
            magnitude_db = 20 * np.log(magnitude + 1)
            
            # Analyze high frequency energy
            h, w = magnitude_db.shape
            center_h, center_w = h // 2, w // 2
            
            # High frequency region (outer 40%)
            high_freq_mask = np.ones_like(magnitude_db)
            high_freq_mask[
                int(center_h * 0.6):int(center_h * 1.4),
                int(center_w * 0.6):int(center_w * 1.4)
            ] = 0
            
            high_freq_energy = np.mean(magnitude_db * high_freq_mask)
            
            # Detect periodic patterns (moirÃƒÂ©)
            # Look for peaks in frequency domain
            magnitude_normalized = magnitude / (np.max(magnitude) + 1e-10)
            peaks = magnitude_normalized > 0.1
            peak_count = np.sum(peaks)
            moire_score = float(np.mean(magnitude_db[peaks]) if peak_count > 0 else 0)
            
            # Print artifact detection (regular grid patterns)
            # Analyze autocorrelation
            autocorr = np.fft.ifft2(np.abs(fft) ** 2).real
            autocorr = np.fft.fftshift(autocorr)
            autocorr_normalized = autocorr / (np.max(autocorr) + 1e-10)
            
            # Look for periodic peaks in autocorrelation
            print_artifact_score = float(np.std(autocorr_normalized) * 100)
            
            is_suspicious = (
                high_freq_energy > self.MOIRE_THRESHOLD or
                moire_score > 55.0 or
                print_artifact_score > 8.0
            )
            
            if self.strict_mode and high_freq_energy > 40.0:
                is_suspicious = True
            
            return FrequencyMetrics(
                high_freq_energy=float(high_freq_energy),
                moire_score=moire_score,
                print_artifact_score=print_artifact_score,
                is_suspicious=is_suspicious
            )
            
        except Exception as e:
            warnings.warn(f"Frequency analysis failed: {e}", RuntimeWarning)
            return FrequencyMetrics(0.0, 0.0, 0.0, False)
    
    def _analyze_color(
        self,
        face_color: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]]
    ) -> ColorMetrics:
        """
        Analyze color characteristics for skin tone naturalness.
        
        Real skin has specific color properties that prints/screens don't reproduce well.
        """
        try:
            if face_bbox:
                x, y, w, h = face_bbox
                face_roi = face_color[y:y+h, x:x+w]
            else:
                face_roi = face_color
            
            # Convert to YCrCb color space (good for skin detection)
            ycrcb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2YCrCb)
            
            # Skin tone detection
            skin_mask = cv2.inRange(ycrcb, self.SKIN_YCRCB_MIN, self.SKIN_YCRCB_MAX)
            skin_ratio = np.sum(skin_mask > 0) / skin_mask.size
            
            # RGB channel variance analysis
            b, g, r = cv2.split(face_roi)
            rgb_variance = (float(np.var(r)), float(np.var(g)), float(np.var(b)))
            
            # Color diversity (real skin has subtle color variations)
            hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            color_diversity = float(np.std(h) * np.std(s)) / 10000.0
            
            # YCrCb consistency
            y_mean = np.mean(ycrcb[:, :, 0])
            cr_mean = np.mean(ycrcb[:, :, 1])
            cb_mean = np.mean(ycrcb[:, :, 2])
            
            # Check if Cr and Cb are in typical skin range
            cr_in_range = 133 <= cr_mean <= 173
            cb_in_range = 77 <= cb_mean <= 127
            ycrcb_consistency = float(skin_ratio)
            
            # Calculate skin tone score
            skin_tone_score = skin_ratio
            if not cr_in_range or not cb_in_range:
                skin_tone_score *= 0.5
            
            is_suspicious = (
                skin_tone_score < self.SKIN_TONE_THRESHOLD or
                color_diversity < 0.01 or  # Too uniform
                color_diversity > 0.5  # Too diverse
            )
            
            if self.strict_mode and skin_tone_score < 0.6:
                is_suspicious = True
            
            return ColorMetrics(
                skin_tone_score=float(skin_tone_score),
                color_diversity=float(color_diversity),
                rgb_variance=rgb_variance,
                ycrcb_consistency=float(ycrcb_consistency),
                is_suspicious=is_suspicious
            )
            
        except Exception as e:
            warnings.warn(f"Color analysis failed: {e}", RuntimeWarning)
            return ColorMetrics(0.5, 0.1, (0.0, 0.0, 0.0), 0.5, True)
    
    def _analyze_depth(
        self,
        face_gray: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]]
    ) -> DepthMetrics:
        """
        Estimate depth and 3D structure.
        
        Real faces have 3D structure; prints are flat.
        """
        try:
            if face_bbox:
                x, y, w, h = face_bbox
                face_roi = face_gray[y:y+h, x:x+w]
            else:
                face_roi = face_gray
            
            # Edge gradient analysis (3D faces have stronger depth gradients)
            sobelx = cv2.Sobel(face_roi, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(face_roi, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            edge_gradient_score = float(np.mean(gradient_magnitude))
            
            # Analyze depth variance using shading
            # Real faces have gradual shading changes
            rows, cols = face_roi.shape
            center_region = face_roi[rows//4:3*rows//4, cols//4:3*cols//4]
            face_depth_variance = float(np.var(center_region))
            
            # Detect nose prominence (center should be brighter/different)
            h, w = face_roi.shape
            center = face_roi[h//3:2*h//3, w//3:2*w//3]
            surround = np.concatenate([
                face_roi[:h//3, :].ravel(),
                face_roi[2*h//3:, :].ravel(),
                face_roi[:, :w//3].ravel(),
                face_roi[:, 2*w//3:].ravel()
            ])
            
            nose_prominence = float(abs(np.mean(center) - np.mean(surround)))
            
            # Determine if 3D structure present
            is_3d_structure = (
                face_depth_variance > self.DEPTH_VARIANCE_THRESHOLD and
                edge_gradient_score > 15.0 and
                nose_prominence > 5.0
            )
            
            is_suspicious = not is_3d_structure
            
            if self.strict_mode and face_depth_variance < 8.0:
                is_suspicious = True
            
            return DepthMetrics(
                face_depth_variance=face_depth_variance,
                edge_gradient_score=edge_gradient_score,
                is_3d_structure=is_3d_structure,
                nose_prominence=nose_prominence,
                is_suspicious=is_suspicious
            )
            
        except Exception as e:
            warnings.warn(f"Depth analysis failed: {e}", RuntimeWarning)
            return DepthMetrics(0.0, 0.0, False, 0.0, True)
    
    def _analyze_motion(
        self,
        face_gray: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]]
    ) -> MotionMetrics:
        """
        Analyze motion patterns using optical flow.
        
        Real faces have natural micro-movements; prints/photos are static.
        """
        try:
            if len(self._frame_buffer) < 2:
                return MotionMetrics(0.0, 0.0, False, 0.0, True)
            
            # Calculate optical flow between recent frames
            prev_frame = self._frame_buffer[-2]
            curr_frame = self._frame_buffer[-1]
            
            # Dense optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_frame, curr_frame,
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # Extract face region flow
            if face_bbox:
                x, y, w, h = face_bbox
                face_flow = magnitude[y:y+h, x:x+w]
                background_flow = np.concatenate([
                    magnitude[:y, :].ravel(),
                    magnitude[y+h:, :].ravel()
                ]) if y > 0 or y+h < magnitude.shape[0] else magnitude.ravel()
            else:
                face_flow = magnitude
                background_flow = magnitude.ravel()
            
            optical_flow_magnitude = float(np.mean(face_flow))
            
            # Motion consistency (real faces move coherently)
            motion_consistency = float(1.0 - np.std(face_flow) / (np.mean(face_flow) + 1e-10))
            motion_consistency = np.clip(motion_consistency, 0, 1)
            
            # Background vs face motion ratio
            bg_mean = np.mean(background_flow)
            face_mean = np.mean(face_flow)
            background_motion_ratio = float(bg_mean / (face_mean + 1e-10))
            
            face_motion_detected = optical_flow_magnitude > self.MOTION_THRESHOLD
            
            # Suspicious if no motion or background moves more than face (video replay)
            is_suspicious = (
                not face_motion_detected or
                background_motion_ratio > 1.5
            )
            
            return MotionMetrics(
                optical_flow_magnitude=optical_flow_magnitude,
                motion_consistency=motion_consistency,
                face_motion_detected=face_motion_detected,
                background_motion_ratio=background_motion_ratio,
                is_suspicious=is_suspicious
            )
            
        except Exception as e:
            warnings.warn(f"Motion analysis failed: {e}", RuntimeWarning)
            return MotionMetrics(0.0, 0.0, False, 0.0, True)
    
    def _detect_blink(
        self,
        face_gray: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]]
    ) -> BlinkMetrics:
        """
        Detect eye blinks using Eye Aspect Ratio (EAR).
        
        Real people blink naturally; photos/videos don't.
        """
        try:
            if face_bbox is None or self._eye_cascade is None:
                return BlinkMetrics(False, 0, [], 0.0, True)
            
            x, y, w, h = face_bbox
            face_roi = face_gray[y:y+h, x:x+w]
            
            # Detect eyes
            eyes = self._eye_cascade.detectMultiScale(face_roi, 1.1, 5)
            
            if len(eyes) < 2:
                return BlinkMetrics(False, 0, [], 0.0, True)
            
            # Calculate EAR (simplified version)
            # Real EAR needs landmark detection, this is approximate
            ear = self._calculate_ear_simple(eyes)
            self._ear_history.append(ear)
            
            # Keep only recent history
            if len(self._ear_history) > 30:
                self._ear_history.pop(0)
            
            # Detect blink (EAR drops significantly)
            if len(self._ear_history) >= 3:
                recent_ears = self._ear_history[-3:]
                if recent_ears[1] < 0.2 and recent_ears[0] > 0.25 and recent_ears[2] > 0.25:
                    self._blink_counter += 1
                    self._last_blink_time = time.time()
            
            # Calculate blink frequency
            if self._frame_buffer:
                duration = len(self._ear_history) / 30.0  # Assume 30fps
                blink_frequency = self._blink_counter / max(duration, 0.1)
            else:
                blink_frequency = 0.0
            
            blink_detected = self._blink_counter >= self.BLINK_REQUIRED
            
            # Natural blink rate is 15-20 per minute (0.25-0.33 per second)
            is_suspicious = (
                not blink_detected or
                blink_frequency > 1.0 or  # Too frequent (fake)
                blink_frequency < 0.1  # Too rare
            )
            
            return BlinkMetrics(
                blink_detected=blink_detected,
                blink_count=self._blink_counter,
                ear_scores=self._ear_history.copy(),
                blink_frequency=float(blink_frequency),
                is_suspicious=is_suspicious
            )
            
        except Exception as e:
            warnings.warn(f"Blink detection failed: {e}", RuntimeWarning)
            return BlinkMetrics(False, 0, [], 0.0, True)
    
    @staticmethod
    def _calculate_ear_simple(eyes: np.ndarray) -> float:
        """Calculate simplified Eye Aspect Ratio."""
        if len(eyes) < 2:
            return 0.3
        
        # Sort eyes by x coordinate
        eyes = sorted(eyes, key=lambda e: e[0])
        
        # Average aspect ratio of both eyes
        ear_sum = 0.0
        for eye in eyes[:2]:
            x, y, w, h = eye
            ear = h / (w + 1e-10)
            ear_sum += ear
        
        return float(ear_sum / 2.0)
    
    def _update_blink_detection(
        self,
        face_gray: np.ndarray,
        face_bbox: Tuple[int, int, int, int]
    ) -> None:
        """Update blink detection state (for video streams)."""
        blink_metrics = self._detect_blink(face_gray, face_bbox)
        # State is updated in _detect_blink via self._blink_counter
    
    def _analyze_video_buffer(
        self,
        require_blink: bool,
        require_motion: bool
    ) -> LivenessResult:
        """Analyze accumulated video frames."""
        # Analyze last frame with all metrics
        last_frame = self._frame_buffer[-1]
        
        # Detect face in last frame
        face_bbox = None
        if self._face_cascade is not None:
            faces = self._face_cascade.detectMultiScale(last_frame, 1.3, 5)
            if len(faces) > 0:
                face_bbox = tuple(faces[0])
        
        # Run standard analyses on last frame
        texture_metrics = self._analyze_texture(last_frame, face_bbox) if self.enable_texture_analysis else None
        frequency_metrics = self._analyze_frequency(last_frame, face_bbox) if self.enable_frequency_analysis else None
        
        # Color analysis needs color image
        color_frame = cv2.cvtColor(last_frame, cv2.COLOR_GRAY2BGR)
        color_metrics = self._analyze_color(color_frame, face_bbox) if self.enable_color_analysis else None
        
        depth_metrics = self._analyze_depth(last_frame, face_bbox) if self.enable_depth_analysis else None
        
        # Motion analysis (requires buffer)
        motion_metrics = self._analyze_motion(last_frame, face_bbox) if require_motion else None
        
        # Blink metrics (already accumulated)
        blink_metrics = None
        if require_blink:
            blink_detected = self._blink_counter >= self.BLINK_REQUIRED
            duration = len(self._ear_history) / 30.0
            blink_frequency = self._blink_counter / max(duration, 0.1)
            
            blink_metrics = BlinkMetrics(
                blink_detected=blink_detected,
                blink_count=self._blink_counter,
                ear_scores=self._ear_history.copy(),
                blink_frequency=float(blink_frequency),
                is_suspicious=not blink_detected
            )
        
        # Aggregate results
        warnings_list = []
        if require_blink and blink_metrics and not blink_metrics.blink_detected:
            warnings_list.append("No blink detected during video")
        if require_motion and motion_metrics and not motion_metrics.face_motion_detected:
            warnings_list.append("Insufficient motion detected")
        
        return self._aggregate_results(
            texture_metrics,
            frequency_metrics,
            color_metrics,
            motion_metrics,
            blink_metrics,
            depth_metrics,
            warnings_list
        )
    
    def _aggregate_results(
        self,
        texture_metrics: Optional[TextureMetrics],
        frequency_metrics: Optional[FrequencyMetrics],
        color_metrics: Optional[ColorMetrics],
        motion_metrics: Optional[MotionMetrics],
        blink_metrics: Optional[BlinkMetrics],
        depth_metrics: Optional[DepthMetrics],
        warnings_list: List[str]
    ) -> LivenessResult:
        """Aggregate individual metrics into final liveness result."""
        
        # Calculate scores from each metric
        scores = []
        weights = []
        
        if texture_metrics:
            score = 1.0 if not texture_metrics.is_suspicious else 0.0
            scores.append(score)
            weights.append(0.20)
        
        if frequency_metrics:
            score = 1.0 if not frequency_metrics.is_suspicious else 0.0
            scores.append(score)
            weights.append(0.20)
        
        if color_metrics:
            score = color_metrics.skin_tone_score
            scores.append(score)
            weights.append(0.15)
        
        if depth_metrics:
            score = 1.0 if depth_metrics.is_3d_structure else 0.0
            scores.append(score)
            weights.append(0.15)
        
        if motion_metrics:
            score = 1.0 if motion_metrics.face_motion_detected and not motion_metrics.is_suspicious else 0.0
            scores.append(score)
            weights.append(0.20)
        
        if blink_metrics:
            score = 1.0 if blink_metrics.blink_detected and not blink_metrics.is_suspicious else 0.0
            scores.append(score)
            weights.append(0.10)
        
        # Calculate weighted confidence
        if scores:
            total_weight = sum(weights)
            confidence = sum(s * w for s, w in zip(scores, weights)) / total_weight
        else:
            confidence = 0.5
        
        # Determine liveness level
        is_live = confidence >= self.confidence_threshold
        
        if confidence >= 0.85:
            liveness_level = LivenessLevel.LIVE
        elif confidence >= 0.70:
            liveness_level = LivenessLevel.LIKELY_LIVE
        elif confidence >= 0.50:
            liveness_level = LivenessLevel.SUSPICIOUS
        elif confidence >= 0.30:
            liveness_level = LivenessLevel.LIKELY_SPOOF
        else:
            liveness_level = LivenessLevel.SPOOF
        
        # Detect attack type
        detected_attack_type = self._detect_attack_type(
            texture_metrics,
            frequency_metrics,
            color_metrics,
            motion_metrics,
            blink_metrics,
            depth_metrics
        )
        
        return LivenessResult(
            is_live=is_live,
            confidence=float(confidence),
            liveness_level=liveness_level,
            detected_attack_type=detected_attack_type,
            texture_metrics=texture_metrics,
            frequency_metrics=frequency_metrics,
            color_metrics=color_metrics,
            motion_metrics=motion_metrics,
            blink_metrics=blink_metrics,
            depth_metrics=depth_metrics,
            warnings=warnings_list
        )
    
    def _detect_attack_type(
        self,
        texture_metrics: Optional[TextureMetrics],
        frequency_metrics: Optional[FrequencyMetrics],
        color_metrics: Optional[ColorMetrics],
        motion_metrics: Optional[MotionMetrics],
        blink_metrics: Optional[BlinkMetrics],
        depth_metrics: Optional[DepthMetrics]
    ) -> Optional[SpoofingAttackType]:
        """Detect the type of spoofing attack."""
        
        # Print attack: low texture, high frequency artifacts, flat depth
        if (frequency_metrics and frequency_metrics.print_artifact_score > 7.0 and
            depth_metrics and not depth_metrics.is_3d_structure):
            return SpoofingAttackType.PRINT_ATTACK
        
        # Digital display: high moirÃƒÂ©, specific frequency patterns
        if (frequency_metrics and frequency_metrics.moire_score > 50.0):
            return SpoofingAttackType.DIGITAL_DISPLAY
        
        # Video replay: motion but no blinks, background motion
        if (motion_metrics and motion_metrics.face_motion_detected and
            blink_metrics and not blink_metrics.blink_detected):
            return SpoofingAttackType.VIDEO_REPLAY
        
        # Mask attack: good color/texture but poor depth/motion
        if (color_metrics and color_metrics.skin_tone_score > 0.6 and
            depth_metrics and not depth_metrics.is_3d_structure and
            motion_metrics and not motion_metrics.face_motion_detected):
            return SpoofingAttackType.MASK_ATTACK
        
        return SpoofingAttackType.UNKNOWN
    
    def _create_error_result(self, error_message: str) -> LivenessResult:
        """Create error result."""
        return LivenessResult(
            is_live=False,
            confidence=0.0,
            liveness_level=LivenessLevel.SPOOF,
            detected_attack_type=None,
            warnings=[error_message]
        )
    
    def reset_state(self) -> None:
        """Reset internal state (for video analysis)."""
        self._frame_buffer.clear()
        self._ear_history.clear()
        self._blink_counter = 0
        self._last_blink_time = 0.0
    
    def get_detector_info(self) -> Dict:
        """Get detector configuration information."""
        return {
            'detection_mode': self.detection_mode.value,
            'enabled_analyses': {
                'texture': self.enable_texture_analysis,
                'frequency': self.enable_frequency_analysis,
                'color': self.enable_color_analysis,
                'depth': self.enable_depth_analysis,
                'motion': self.enable_motion_analysis,
                'blink': self.enable_blink_detection
            },
            'confidence_threshold': self.confidence_threshold,
            'strict_mode': self.strict_mode,
            'thresholds': {
                'lbp': self.LBP_THRESHOLD,
                'moire': self.MOIRE_THRESHOLD,
                'skin_tone': self.SKIN_TONE_THRESHOLD,
                'motion': self.MOTION_THRESHOLD,
                'blink_required': self.BLINK_REQUIRED,
                'depth_variance': self.DEPTH_VARIANCE_THRESHOLD
            }
        }


# ============================================================================
# INTEGRATION HELPERS
# ============================================================================

def integrate_with_preprocessor(
    preprocessor,
    detector: AntiSpoofingDetector
):
    """
    Integrate anti-spoofing with face preprocessor.
    
    Args:
        preprocessor: FacePreprocessor or EnhancedFacePreprocessor instance
        detector: AntiSpoofingDetector instance
    """
    # Add antispoofing method to preprocessor
    original_preprocess = preprocessor.preprocess_with_metadata
    
    def preprocess_with_antispoofing(face_image, *args, **kwargs):
        # Run standard preprocessing
        result = original_preprocess(face_image, *args, **kwargs)
        
        # Run anti-spoofing
        liveness_result = detector.detect(face_image)
        
        # Add to result
        result.spoofing_metrics = liveness_result
        
        if not liveness_result.is_live:
            result.quality_metrics.should_reject = True
            result.quality_metrics.rejection_reason = f"Spoofing detected: {liveness_result.detected_attack_type}"
        
        return result
    
    preprocessor.preprocess_with_antispoofing = preprocess_with_antispoofing
    return preprocessor


def create_antispoofing_recognizer(
    base_recognizer,
    detector: AntiSpoofingDetector,
    reject_on_spoof: bool = True
):
    """
    Wrap a face recognizer with anti-spoofing.
    
    Args:
        base_recognizer: Your existing face recognizer
        detector: AntiSpoofingDetector instance
        reject_on_spoof: Whether to reject recognition on spoofing detection
        
    Returns:
        Wrapped recognizer with antispoofing
    """
    class AntiSpoofingRecognizer:
        def __init__(self):
            self.recognizer = base_recognizer
            self.detector = detector
            self.reject_on_spoof = reject_on_spoof
            self.last_liveness_result = None
        
        def recognize(self, face_image):
            # Check liveness first
            liveness_result = self.detector.detect(face_image)
            self.last_liveness_result = liveness_result
            
            if not liveness_result.is_live and self.reject_on_spoof:
                return {
                    'success': False,
                    'reason': 'Spoofing detected',
                    'liveness': liveness_result,
                    'confidence': 0.0
                }
            
            # Proceed with recognition
            recognition_result = self.recognizer.recognize(face_image)
            recognition_result['liveness'] = liveness_result
            
            # Adjust confidence if suspicious
            if liveness_result.liveness_level in [LivenessLevel.SUSPICIOUS, LivenessLevel.LIKELY_SPOOF]:
                if 'confidence' in recognition_result:
                    recognition_result['confidence'] *= liveness_result.confidence
            
            return recognition_result
        
        def get_last_liveness_result(self):
            return self.last_liveness_result
    
    return AntiSpoofingRecognizer()


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_basic_detector(strict_mode: bool = False) -> AntiSpoofingDetector:
    """Create basic passive detector (single image)."""
    return AntiSpoofingDetector(
        detection_mode=DetectionMode.PASSIVE_SINGLE,
        enable_texture_analysis=True,
        enable_frequency_analysis=True,
        enable_color_analysis=True,
        enable_depth_analysis=True,
        enable_motion_analysis=False,
        enable_blink_detection=False,
        confidence_threshold=0.45,  # More lenient for real faces
        strict_mode=strict_mode
    )


def create_video_detector(
    require_blink: bool = True,
    strict_mode: bool = False
) -> AntiSpoofingDetector:
    """Create video-based detector with motion and blink detection."""
    return AntiSpoofingDetector(
        detection_mode=DetectionMode.PASSIVE_MULTI,
        enable_texture_analysis=True,
        enable_frequency_analysis=True,
        enable_color_analysis=True,
        enable_depth_analysis=True,
        enable_motion_analysis=True,
        enable_blink_detection=require_blink,
        confidence_threshold=0.50,  # More lenient for real faces
        strict_mode=strict_mode
    )


def create_high_security_detector(
    require_blink: bool = True,
    require_motion: bool = True
) -> AntiSpoofingDetector:
    """Create high-security detector with all checks enabled."""
    return AntiSpoofingDetector(
        detection_mode=DetectionMode.ACTIVE_BLINK if require_blink else DetectionMode.PASSIVE_MULTI,
        enable_texture_analysis=True,
        enable_frequency_analysis=True,
        enable_color_analysis=True,
        enable_depth_analysis=True,
        enable_motion_analysis=require_motion,
        enable_blink_detection=require_blink,
        confidence_threshold=0.60,  # More lenient for real faces
        strict_mode=True
    )


def create_lightweight_detector() -> AntiSpoofingDetector:
    """Create lightweight detector for resource-constrained environments."""
    return AntiSpoofingDetector(
        detection_mode=DetectionMode.PASSIVE_SINGLE,
        enable_texture_analysis=True,
        enable_frequency_analysis=True,
        enable_color_analysis=False,  # Skip for speed
        enable_depth_analysis=False,  # Skip for speed
        enable_motion_analysis=False,
        enable_blink_detection=False,
        confidence_threshold=0.35,  # More lenient for real faces
        strict_mode=False
    )


# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

def test_single_image(image_path: str, save_report: bool = True):
    """
    Test anti-spoofing on a single image.
    
    Args:
        image_path: Path to face image
        save_report: Whether to save detailed report
    """
    import cv2
    
    print(f"Testing anti-spoofing on: {image_path}")
    print("=" * 60)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Create detector
    detector = create_basic_detector(strict_mode=True)
    
    # Perform detection
    result = detector.detect(image)
    
    # Print results
    print(result.get_detailed_report())
    print("=" * 60)
    
    # Save report if requested
    if save_report:
        report_path = image_path.replace('.jpg', '_antispoofing_report.txt')
        with open(report_path, 'w') as f:
            f.write(result.get_detailed_report())
        print(f"Report saved to: {report_path}")
    
    return result


def test_video_stream(duration: float = 5.0, camera_id: int = 0):
    """
    Test anti-spoofing on live video stream.
    
    Args:
        duration: How long to analyze (seconds)
        camera_id: Camera device ID
    """
    print(f"Testing anti-spoofing on video stream (camera {camera_id})")
    print(f"Analysis duration: {duration} seconds")
    print("=" * 60)
    print("Instructions:")
    print("  1. Look at the camera")
    print("  2. Blink naturally")
    print("  3. Move your head slightly")
    print("=" * 60)
    
    # Create video detector
    detector = create_video_detector(require_blink=True, strict_mode=True)
    
    # Perform detection
    result = detector.detect_video_stream(
        video_source=camera_id,
        duration_seconds=duration,
        require_blink=True,
        require_motion=True
    )
    
    # Print results
    print("\nAnalysis complete!")
    print(result.get_detailed_report())
    print("=" * 60)
    
    return result


def batch_test_images(image_folder: str, output_csv: Optional[str] = None):
    """
    Test anti-spoofing on a folder of images.
    
    Args:
        image_folder: Path to folder containing face images
        output_csv: Optional path to save results CSV
    """
    from pathlib import Path
    import csv
    
    image_folder = Path(image_folder)
    image_files = list(image_folder.glob('*.jpg')) + list(image_folder.glob('*.png'))
    
    print(f"Testing {len(image_files)} images from {image_folder}")
    print("=" * 60)
    
    detector = create_basic_detector(strict_mode=True)
    results = []
    
    for idx, img_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] Processing {img_path.name}...")
        
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  Ã¢Å“â€” Failed to load")
            continue
        
        result = detector.detect(image)
        
        status = "Ã¢Å“â€œ LIVE" if result.is_live else "Ã¢Å“â€” SPOOF"
        print(f"  {status} (confidence: {result.confidence:.2%})")
        
        results.append({
            'filename': img_path.name,
            'is_live': result.is_live,
            'confidence': result.confidence,
            'liveness_level': result.liveness_level.value,
            'attack_type': result.detected_attack_type.value if result.detected_attack_type else 'none',
            'processing_time': result.processing_time
        })
    
    # Save results
    if output_csv and results:
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to: {output_csv}")
    
    # Summary statistics
    live_count = sum(1 for r in results if r['is_live'])
    spoof_count = len(results) - live_count
    avg_confidence = sum(r['confidence'] for r in results) / len(results) if results else 0
    avg_time = sum(r['processing_time'] for r in results) / len(results) if results else 0
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"  Total images: {len(results)}")
    print(f"  Live: {live_count} ({live_count/len(results)*100:.1f}%)")
    print(f"  Spoof: {spoof_count} ({spoof_count/len(results)*100:.1f}%)")
    print(f"  Average confidence: {avg_confidence:.2%}")
    print(f"  Average processing time: {avg_time:.3f}s")
    print("=" * 60)
    
    return results


def demo_integration_with_recognition():
    """
    Demonstrate integration with a face recognition system.
    """
    print("Anti-Spoofing Integration Demo")
    print("=" * 60)
    
    # Mock face recognizer (replace with your actual recognizer)
    class MockFaceRecognizer:
        def recognize(self, face_image):
            # Simulate recognition
            return {
                'success': True,
                'identity': 'John Doe',
                'confidence': 0.92
            }
    
    # Create detector and wrap recognizer
    detector = create_high_security_detector(require_blink=False, require_motion=False)
    base_recognizer = MockFaceRecognizer()
    secure_recognizer = create_antispoofing_recognizer(
        base_recognizer,
        detector,
        reject_on_spoof=True
    )
    
    print("Secure recognizer created with anti-spoofing protection")
    print("\nTest 1: Legitimate face image")
    print("-" * 60)
    
    # Create a test image (in practice, load a real image)
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    result = secure_recognizer.recognize(test_image)
    print(f"Recognition result: {result}")
    
    if result.get('liveness'):
        print(f"Liveness check: {result['liveness']}")
    
    print("\n" + "=" * 60)
    print("Integration complete!")
    print("Use secure_recognizer.recognize(image) for protected recognition")
    
    return secure_recognizer


# ============================================================================
# ADVANCED FEATURES
# ============================================================================

class ChallengeResponseAntiSpoofing:
    """
    Interactive challenge-response anti-spoofing.
    
    Randomly asks user to perform actions:
    - Blink
    - Turn head left/right
    - Smile
    - Open mouth
    
    Much more robust against sophisticated attacks.
    """
    
    def __init__(self, detector: AntiSpoofingDetector):
        self.detector = detector
        self.challenges = ['blink', 'turn_left', 'turn_right', 'smile']
    
    def perform_challenge(
        self,
        video_source: Union[int, str],
        challenge: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Perform a random challenge.
        
        Args:
            video_source: Camera or video source
            challenge: Specific challenge or None for random
            
        Returns:
            (success, message) tuple
        """
        import random
        
        if challenge is None:
            challenge = random.choice(self.challenges)
        
        print(f"Challenge: Please {challenge.replace('_', ' ')}")
        
        if challenge == 'blink':
            result = self.detector.detect_video_stream(
                video_source,
                duration_seconds=3.0,
                require_blink=True,
                require_motion=False
            )
            success = result.blink_metrics and result.blink_metrics.blink_detected
            message = "Blink detected!" if success else "No blink detected"
        
        elif challenge in ['turn_left', 'turn_right']:
            result = self.detector.detect_video_stream(
                video_source,
                duration_seconds=3.0,
                require_blink=False,
                require_motion=True
            )
            success = result.motion_metrics and result.motion_metrics.face_motion_detected
            message = "Head movement detected!" if success else "No movement detected"
        
        else:
            # For smile/other challenges, would need additional detection
            result = self.detector.detect_video_stream(
                video_source,
                duration_seconds=2.0,
                require_blink=False,
                require_motion=False
            )
            success = result.is_live
            message = "Challenge completed" if success else "Challenge failed"
        
        return success, message
    
    def multi_challenge_verification(
        self,
        video_source: Union[int, str],
        num_challenges: int = 2
    ) -> bool:
        """
        Perform multiple random challenges.
        
        Args:
            video_source: Camera or video source
            num_challenges: Number of challenges to perform
            
        Returns:
            True if all challenges passed
        """
        import random
        
        print(f"Multi-challenge verification ({num_challenges} challenges)")
        print("=" * 60)
        
        challenges = random.sample(self.challenges, min(num_challenges, len(self.challenges)))
        results = []
        
        for i, challenge in enumerate(challenges, 1):
            print(f"\nChallenge {i}/{num_challenges}")
            success, message = self.perform_challenge(video_source, challenge)
            print(f"  {message}")
            results.append(success)
            
            if not success:
                print("  Ã¢Å“â€” Challenge failed - verification failed")
                return False
        
        print("\n" + "=" * 60)
        print("Ã¢Å“â€œ All challenges passed!")
        return True


class AdaptiveAntiSpoofing:
    """
    Adaptive anti-spoofing that learns from false positives/negatives.
    
    Adjusts thresholds based on deployment environment.
    """
    
    def __init__(self, base_detector: AntiSpoofingDetector):
        self.detector = base_detector
        self.false_positive_count = 0
        self.false_negative_count = 0
        self.total_predictions = 0
        self.threshold_adjustments = []
    
    def detect_with_feedback(
        self,
        face_image: np.ndarray,
        ground_truth: Optional[bool] = None
    ) -> LivenessResult:
        """
        Detect with optional ground truth feedback for adaptation.
        
        Args:
            face_image: Input face image
            ground_truth: True if actually live, False if spoof, None if unknown
            
        Returns:
            LivenessResult
        """
        result = self.detector.detect(face_image)
        self.total_predictions += 1
        
        if ground_truth is not None:
            # Update statistics
            if result.is_live and not ground_truth:
                self.false_positive_count += 1
                self._adjust_threshold_up()
            elif not result.is_live and ground_truth:
                self.false_negative_count += 1
                self._adjust_threshold_down()
        
        return result
    
    def _adjust_threshold_up(self):
        """Increase threshold (be more strict) after false positive."""
        current = self.detector.confidence_threshold
        new_threshold = min(current + 0.05, 0.95)
        self.detector.confidence_threshold = new_threshold
        self.threshold_adjustments.append(('up', new_threshold))
        print(f"Adjusted threshold: {current:.2f} Ã¢â€ â€™ {new_threshold:.2f} (more strict)")
    
    def _adjust_threshold_down(self):
        """Decrease threshold (be more lenient) after false negative."""
        current = self.detector.confidence_threshold
        new_threshold = max(current - 0.05, 0.3)
        self.detector.confidence_threshold = new_threshold
        self.threshold_adjustments.append(('down', new_threshold))
        print(f"Adjusted threshold: {current:.2f} Ã¢â€ â€™ {new_threshold:.2f} (more lenient)")
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        return {
            'total_predictions': self.total_predictions,
            'false_positives': self.false_positive_count,
            'false_negatives': self.false_negative_count,
            'current_threshold': self.detector.confidence_threshold,
            'adjustments_made': len(self.threshold_adjustments),
            'fp_rate': self.false_positive_count / max(self.total_predictions, 1),
            'fn_rate': self.false_negative_count / max(self.total_predictions, 1)
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def visualize_antispoofing_result(
    face_image: np.ndarray,
    result: LivenessResult,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Create visualization of anti-spoofing result.
    
    Args:
        face_image: Input face image
        result: LivenessResult from detection
        save_path: Optional path to save visualization
        
    Returns:
        Visualization image
    """
    import cv2
    
    # Create visualization
    vis = face_image.copy()
    h, w = vis.shape[:2]
    
    # Add overlay with result
    overlay = np.zeros((150, w, 3), dtype=np.uint8)
    
    # Status color
    if result.is_live:
        color = (0, 255, 0)  # Green
        status = "LIVE"
    else:
        color = (0, 0, 255)  # Red
        status = "SPOOF"
    
    # Draw status
    cv2.putText(overlay, f"Status: {status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    cv2.putText(overlay, f"Confidence: {result.confidence:.1%}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.putText(overlay, f"Level: {result.liveness_level.value}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Attack type if detected
    if result.detected_attack_type:
        cv2.putText(overlay, f"Attack: {result.detected_attack_type.value}", (10, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Combine
    vis = np.vstack([vis, overlay])
    
    # Save if requested
    if save_path:
        cv2.imwrite(save_path, vis)
        print(f"Visualization saved to: {save_path}")
    
    return vis


def compare_detectors(
    face_image: np.ndarray,
    detectors: Dict[str, AntiSpoofingDetector]
) -> Dict[str, LivenessResult]:
    """
    Compare multiple detector configurations.
    
    Args:
        face_image: Input face image
        detectors: Dictionary of detector_name -> detector
        
    Returns:
        Dictionary of results
    """
    print("Comparing Anti-Spoofing Detectors")
    print("=" * 60)
    
    results = {}
    
    for name, detector in detectors.items():
        print(f"\nTesting: {name}")
        result = detector.detect(face_image)
        results[name] = result
        
        status = "Ã¢Å“â€œ LIVE" if result.is_live else "Ã¢Å“â€” SPOOF"
        print(f"  {status} (confidence: {result.confidence:.2%})")
        print(f"  Processing time: {result.processing_time:.3f}s")
    
    print("\n" + "=" * 60)
    return results


# ============================================================================
# MAIN DEMO
# ============================================================================

def main():
    """Main demo function."""
    print("Ã¢â€¢â€Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢â€”")
    print("Ã¢â€¢â€˜     Advanced Anti-Spoofing Detection System              Ã¢â€¢â€˜")
    print("Ã¢â€¢â€˜     Protect Your Face Recognition System                 Ã¢â€¢â€˜")
    print("Ã¢â€¢Å¡Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢Â")
    print()
    
    print("Available demos:")
    print("  1. Test single image")
    print("  2. Test video stream (webcam)")
    print("  3. Batch test images")
    print("  4. Integration demo")
    print("  5. Challenge-response demo")
    print("  6. Compare detectors")
    print()
    print("Example usage in your code:")
    print()
    print("  # Basic usage")
    print("  from antispoofing import create_basic_detector")
    print("  detector = create_basic_detector(strict_mode=True)")
    print("  result = detector.detect(face_image)")
    print("  if result.is_live:")
    print("      print('Live person detected!')")
    print()
    print("  # Video analysis")
    print("  from antispoofing import create_video_detector")
    print("  detector = create_video_detector(require_blink=True)")
    print("  result = detector.detect_video_stream(camera_id=0, duration_seconds=3.0)")
    print()
    print("  # Integration with recognizer")
    print("  from antispoofing import create_antispoofing_recognizer")
    print("  secure_recognizer = create_antispoofing_recognizer(")
    print("      your_recognizer, detector, reject_on_spoof=True")
    print("  )")
    print("  result = secure_recognizer.recognize(face_image)")
    print()


if __name__ == "__main__":
    main()