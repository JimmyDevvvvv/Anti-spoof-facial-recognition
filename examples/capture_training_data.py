#!/usr/bin/env python3
"""Training Data Capture Script

Professional-grade training data capture for face recognition systems.
Features automatic quality validation, intelligent capture timing, and
comprehensive progress tracking with advanced image quality controls.

Usage:
    python examples/capture_training_data.py --name "John Doe" --samples 30
    python examples/capture_training_data.py --name "Jane Smith" --samples 20 --camera 1
    python examples/capture_training_data.py --name "Alice" --mode manual --quality-threshold 0.7
"""

import argparse
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Fix Windows terminal encoding for emojis
if sys.platform == "win32":
    try:
        import codecs
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, errors='replace')
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, errors='replace')
    except Exception:
        pass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.face.detector import FaceDetector
    from src.face.preprocessing import FacePreprocessor
except ImportError as e:
    print(f"❌ Error: Could not import required modules: {e}")
    print("   Make sure you're running from the project root directory")
    sys.exit(1)


class CaptureMode(Enum):
    """Capture mode options."""
    AUTO = "auto"           # Automatic capture
    MANUAL = "manual"       # Manual capture with spacebar
    HYBRID = "hybrid"       # Both auto and manual


class CaptureStatus(Enum):
    """Capture session status."""
    INITIALIZING = "initializing"
    READY = "ready"
    CAPTURING = "capturing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class CaptureStats:
    """Statistics for a capture session."""
    total_captured: int = 0
    quality_failures: int = 0
    multiple_faces: int = 0
    no_face_detected: int = 0
    session_duration: float = 0.0
    avg_quality_score: float = 0.0
    min_quality_score: float = 1.0
    max_quality_score: float = 0.0
    quality_scores: List[float] = field(default_factory=list)
    capture_timestamps: List[float] = field(default_factory=list)
    face_size_history: List[Tuple[int, int]] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate capture success rate."""
        total_attempts = (
            self.total_captured + 
            self.quality_failures + 
            self.multiple_faces
        )
        return (self.total_captured / total_attempts * 100) if total_attempts > 0 else 0.0
    
    @property
    def avg_face_size(self) -> Tuple[float, float]:
        """Calculate average face size."""
        if not self.face_size_history:
            return (0.0, 0.0)
        avg_w = np.mean([w for w, h in self.face_size_history])
        avg_h = np.mean([h for w, h in self.face_size_history])
        return (avg_w, avg_h)
    
    @property
    def captures_per_minute(self) -> float:
        """Calculate capture rate."""
        if self.session_duration <= 0:
            return 0.0
        return (self.total_captured / self.session_duration) * 60
    
    def update_quality_stats(self, quality: float):
        """Update quality statistics."""
        self.quality_scores.append(quality)
        self.min_quality_score = min(self.min_quality_score, quality)
        self.max_quality_score = max(self.max_quality_score, quality)
        self.avg_quality_score = np.mean(self.quality_scores)
    
    def __str__(self) -> str:
        """Human-readable statistics."""
        avg_w, avg_h = self.avg_face_size
        return (
            f"Capture Statistics:\n"
            f"  ✓ Successfully captured: {self.total_captured}\n"
            f"  ✗ Quality failures: {self.quality_failures}\n"
            f"  ⚠ Multiple faces: {self.multiple_faces}\n"
            f"  ⚠ No face detected: {self.no_face_detected}\n"
            f"  ⏱ Session duration: {self.session_duration:.1f}s\n"
            f"  ⚡ Capture rate: {self.captures_per_minute:.1f} per minute\n"
            f"  📊 Success rate: {self.success_rate:.1f}%\n"
            f"  ⭐ Avg quality: {self.avg_quality_score:.2f} "
            f"(min: {self.min_quality_score:.2f}, max: {self.max_quality_score:.2f})\n"
            f"  📐 Avg face size: {avg_w:.0f}x{avg_h:.0f} pixels"
        )


class TrainingDataCapture:
    """
    Professional training data capture system with quality control.
    
    Features:
    - High-resolution capture (1080p)
    - Automatic face detection with quality validation
    - Intelligent capture timing (avoids duplicates)
    - Real-time quality assessment and feedback
    - Progress tracking and detailed statistics
    - Visual feedback with corner markers
    - Multiple capture modes (auto/manual/hybrid)
    - Diversity tracking (pose, expression variation)
    """
    
    # Display colors (BGR format)
    COLOR_SUCCESS = (0, 255, 0)      # Green
    COLOR_WARNING = (0, 165, 255)    # Orange
    COLOR_ERROR = (0, 0, 255)        # Red
    COLOR_INFO = (255, 255, 0)       # Cyan
    COLOR_WHITE = (255, 255, 255)    # White
    COLOR_DARK_BG = (40, 40, 40)     # Dark gray
    
    # Timing constants
    MIN_CAPTURE_DELAY = 0.5          # Minimum seconds between captures (increased)
    AUTO_CAPTURE_FRAMES = 20         # Frames between auto-captures (increased)
    FLASH_DURATION_MS = 150          # Flash effect duration
    
    # Quality thresholds
    DEFAULT_MIN_QUALITY = 0.6        # Minimum quality score (increased)
    EXCELLENT_QUALITY = 0.8          # Excellent quality threshold
    
    # Face size requirements
    MIN_FACE_SIZE = 100              # Minimum face size in pixels
    TARGET_FACE_SIZE = 300           # Target face size for optimal quality
    MAX_FACE_SIZE = 600              # Maximum face size (too close)
    
    # Resolution settings
    CAMERA_WIDTH = 1920              # 1080p resolution
    CAMERA_HEIGHT = 1080
    TARGET_FACE_RESOLUTION = (150, 150)  # Higher resolution for saved faces

    def __init__(
        self,
        person_name: str,
        num_samples: int = 30,
        camera_index: int = 0,
        output_dir: str = "training_data",
        mode: CaptureMode = CaptureMode.AUTO,
        min_quality: float = DEFAULT_MIN_QUALITY,
        use_clahe: bool = True,
        enable_alignment: bool = False,
        save_original: bool = False,
        enable_diversity_check: bool = True,
    ):
        """
        Initialize the training data capture system.
        
        Args:
            person_name: Name of the person to capture
            num_samples: Number of samples to capture
            camera_index: Camera device index
            output_dir: Directory to save training images
            mode: Capture mode (auto/manual/hybrid)
            min_quality: Minimum quality score (0.0-1.0)
            use_clahe: Use CLAHE preprocessing
            enable_alignment: Enable face alignment (slower but better quality)
            save_original: Save original unprocessed images alongside processed
            enable_diversity_check: Check for pose/expression diversity
        """
        self.person_name = person_name
        self.num_samples = num_samples
        self.camera_index = camera_index
        self.output_dir = Path(output_dir)
        self.mode = mode
        self.min_quality = min_quality
        self.save_original = save_original
        self.enable_diversity = enable_diversity_check
        
        # Create output directory
        self.person_dir = self.output_dir / self._sanitize_name(person_name)
        self.person_dir.mkdir(parents=True, exist_ok=True)
        
        if save_original:
            self.original_dir = self.person_dir / "originals"
            self.original_dir.mkdir(exist_ok=True)
        
        # Initialize components with optimized parameters
        self.detector = FaceDetector(
            scale_factor=1.1,        # Better detection
            min_neighbors=5,         # More sensitive
            min_size=(self.MIN_FACE_SIZE, self.MIN_FACE_SIZE),
            use_clahe=use_clahe
        )
    
        self.preprocessor = FacePreprocessor(
            target_size=self.TARGET_FACE_RESOLUTION,
            use_clahe=use_clahe,
            enable_alignment=enable_alignment
        )
        
        # Session state
        self.status = CaptureStatus.INITIALIZING
        self.stats = CaptureStats()
        self.captured_count = 0
        self.last_capture_time = 0.0
        self.session_start_time = 0.0
        self.frame_count = 0
        self.last_capture_face_center: Optional[Tuple[float, float]] = None
        
        # Camera
        self.cap: Optional[cv2.VideoCapture] = None
        
        # Diversity tracking
        self.captured_face_centers: List[Tuple[float, float]] = []
        self.diversity_threshold = 0.05  # 5% of frame size

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Sanitize person name for directory/file names."""
        # Remove or replace problematic characters
        sanitized = name.replace(" ", "_")
        for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
            sanitized = sanitized.replace(char, '_')
        return sanitized

    def initialize_camera(self) -> bool:
        """
        Initialize camera with optimal settings.
        
        Returns:
            True if camera initialized successfully
        """
        print("🎥 Initializing camera...")
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print(f"❌ Error: Could not open camera {self.camera_index}")
            print("   Try different camera index: --camera 1 or --camera 2")
            self.status = CaptureStatus.FAILED
            return False
        
        # Set camera properties for best quality (1080p)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Enable auto-adjustments
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        
        # Verify actual resolution
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"✓ Camera initialized: {actual_width}x{actual_height}")
        
        if actual_width < self.CAMERA_WIDTH or actual_height < self.CAMERA_HEIGHT:
            print(f"⚠️  Warning: Requested {self.CAMERA_WIDTH}x{self.CAMERA_HEIGHT} "
                  f"but got {actual_width}x{actual_height}")
            print("   Capture will continue with available resolution")
        
        # Warm up camera
        print("⏳ Warming up camera (3 seconds)...")
        for _ in range(90):  # 3 seconds at 30fps
            self.cap.read()
        
        self.status = CaptureStatus.READY
        return True

    def print_instructions(self) -> None:
        """Print capture instructions to console."""
        print(f"\n{'='*70}")
        print(f"📸 TRAINING DATA CAPTURE SESSION")
        print(f"{'='*70}")
        print(f"Person:            {self.person_name}")
        print(f"Target samples:    {self.num_samples}")
        print(f"Capture mode:      {self.mode.value.upper()}")
        print(f"Quality threshold: {self.min_quality:.2f}")
        print(f"Face resolution:   {self.TARGET_FACE_RESOLUTION[0]}x{self.TARGET_FACE_RESOLUTION[1]}")
        print(f"Min face size:     {self.MIN_FACE_SIZE}x{self.MIN_FACE_SIZE} pixels")
        print(f"Output directory:  {self.person_dir}")
        print(f"{'='*70}")
        
        print("\n📋 BEST PRACTICES:")
        print("  ✓ Position face 2-3 feet from camera")
        print("  ✓ Ensure face fills 20-40% of frame")
        print("  ✓ Keep face well-lit with even lighting")
        print("  ✓ Avoid backlighting (light behind you)")
        print("  ✓ Look directly at camera for most shots")
        print("  ✓ Gradually vary head position:")
        print("    - Turn left/right (±30°)")
        print("    - Tilt up/down (±20°)")
        print("    - Slight forward/backward movement")
        print("  ✓ Try different expressions:")
        print("    - Neutral (most important)")
        print("    - Slight smile")
        print("    - Serious/focused")
        print("  ✓ Keep glasses/accessories consistent")
        
        print("\n⌨️  CONTROLS:")
        if self.mode == CaptureMode.AUTO:
            print("  • AUTO MODE: Images capture automatically")
            print("    - Green border = Ready to capture")
            print("    - Orange border = Adjusting quality")
        elif self.mode == CaptureMode.MANUAL:
            print("  • MANUAL MODE: Press SPACE to capture")
        else:  # HYBRID
            print("  • HYBRID MODE:")
            print("    - Press SPACE for manual capture")
            print("    - Or wait for automatic capture")
        
        print("\n  • Press P to pause/resume")
        print("  • Press D to display statistics")
        print("  • Press ESC or Q to finish")
        print(f"{'='*70}\n")
        
        input("Press ENTER when ready to start...")
        print()

    def check_diversity(self, face_center: Tuple[float, float]) -> Tuple[bool, str]:
        """
        Check if captured face adds diversity to dataset.
        
        Args:
            face_center: Normalized (x, y) center of face (0.0-1.0)
            
        Returns:
            Tuple of (is_diverse, message)
        """
        if not self.enable_diversity or len(self.captured_face_centers) < 5:
            return True, "OK"
        
        # Check if this position is too similar to recent captures
        for prev_center in self.captured_face_centers[-5:]:
            distance = np.sqrt(
                (face_center[0] - prev_center[0])**2 +
                (face_center[1] - prev_center[1])**2
            )
            
            if distance < self.diversity_threshold:
                return False, "Too similar to recent capture - Move your head"
        
        return True, "OK"

    def assess_face_size(
        self,
        face_bbox: Tuple[int, int, int, int],
        frame_shape: Tuple[int, int]
    ) -> Tuple[bool, str]:
        """
        Assess if face size is appropriate.
        
        Args:
            face_bbox: Face bounding box (x, y, w, h)
            frame_shape: Frame shape (height, width)
            
        Returns:
            Tuple of (is_good_size, message)
        """
        x, y, w, h = face_bbox
        frame_h, frame_w = frame_shape[:2]
        
        # Calculate face size ratio
        face_area = w * h
        frame_area = frame_w * frame_h
        size_ratio = face_area / frame_area
        
        if w < self.MIN_FACE_SIZE or h < self.MIN_FACE_SIZE:
            return False, f"Face too small ({w}x{h}px) - Move closer"
        
        if w > self.MAX_FACE_SIZE or h > self.MAX_FACE_SIZE:
            return False, f"Face too large ({w}x{h}px) - Move back"
        
        if size_ratio < 0.03:
            return False, "Face too small - Move closer to camera"
        
        if size_ratio > 0.5:
            return False, "Face too large - Move back from camera"
        
        return True, f"Good size: {w}x{h}px ({size_ratio*100:.1f}% of frame)"

    def assess_capture_readiness(
        self,
        frame: np.ndarray,
        faces: List[Tuple[int, int, int, int]]
    ) -> Tuple[bool, str, Optional[Tuple[int, int, int, int]], Optional[float]]:
        """
        Assess if frame is ready for capture with comprehensive checks.
        
        Returns:
            Tuple of (ready, message, face_bbox, quality_score)
        """
        frame_h, frame_w = frame.shape[:2]
        
        # No face detected
        if len(faces) == 0:
            self.stats.no_face_detected += 1
            return False, "No face detected - Position yourself in frame", None, None
        
        # Multiple faces detected
        if len(faces) > 1:
            self.stats.multiple_faces += 1
            return (
                False,
                f"Multiple faces ({len(faces)}) - Ensure only you are visible",
                None,
                None
            )
        
        # Single face - comprehensive validation
        x, y, w, h = faces[0]
        
        # Check face size
        size_ok, size_msg = self.assess_face_size(faces[0], frame.shape)
        if not size_ok:
            return False, size_msg, faces[0], None
        
        # Extract face ROI
        face_roi = frame[y:y+h, x:x+w]
        
        # Validate face quality
        validation = self.detector.validate_face(
            face_roi,
            min_quality_threshold=self.min_quality
        )
        
        if not validation.is_valid:
            self.stats.quality_failures += 1
            reasons = ", ".join(validation.recommendations[:2])
            return False, f"Quality: {reasons}", faces[0], validation.quality_score
        
        # Check diversity (pose variation)
        face_center_x = (x + w/2) / frame_w
        face_center_y = (y + h/2) / frame_h
        diverse, diversity_msg = self.check_diversity((face_center_x, face_center_y))
        
        if not diverse:
            return False, diversity_msg, faces[0], validation.quality_score
        
        # All checks passed
        return True, size_msg, faces[0], validation.quality_score

    def should_auto_capture(self) -> bool:
        """Check if auto-capture should trigger."""
        if self.mode == CaptureMode.MANUAL:
            return False
        
        current_time = time.time()
        time_since_last = current_time - self.last_capture_time
        
        # Check timing and frame count
        return (
            self.frame_count % self.AUTO_CAPTURE_FRAMES == 0 and
            time_since_last > self.MIN_CAPTURE_DELAY
        )

    def capture_face(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        quality_score: float
    ) -> bool:
        """
        Capture and save a face image with optional original.
        
        Returns:
            True if capture successful
        """
        try:
            x, y, w, h = bbox
            face_img = frame[y:y+h, x:x+w].copy()
            
            # Save original if requested
            if self.save_original:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                orig_filename = f"{self._sanitize_name(self.person_name)}_orig_{self.captured_count:03d}_{timestamp}.jpg"
                orig_path = self.original_dir / orig_filename
                cv2.imwrite(str(orig_path), face_img)
            
            # Preprocess face
            preprocessed = self.preprocessor.preprocess(
                face_img,
                align=False,  # Alignment can be enabled for higher quality
                equalize=True
            )
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{self._sanitize_name(self.person_name)}_{self.captured_count:03d}_{timestamp}.jpg"
            img_path = self.person_dir / filename
            
            # Save processed image
            success = cv2.imwrite(str(img_path), preprocessed)
            
            if not success:
                warnings.warn(f"Failed to save image: {img_path}", RuntimeWarning)
                return False
            
            # Update stats
            self.captured_count += 1
            self.stats.total_captured += 1
            self.stats.update_quality_stats(quality_score)
            self.stats.capture_timestamps.append(time.time())
            self.stats.face_size_history.append((w, h))
            
            # Update diversity tracking
            frame_h, frame_w = frame.shape[:2]
            face_center_x = (x + w/2) / frame_w
            face_center_y = (y + h/2) / frame_h
            self.captured_face_centers.append((face_center_x, face_center_y))
            
            self.last_capture_time = time.time()
            
            # Log capture with enhanced info
            quality_stars = "⭐" * int(quality_score * 5)
            progress_pct = (self.captured_count / self.num_samples) * 100
            print(f"✓ [{self.captured_count:3d}/{self.num_samples}] ({progress_pct:5.1f}%) "
                  f"Quality: {quality_score:.2f} {quality_stars} | "
                  f"Size: {w}x{h}px | "
                  f"Saved: {filename}")
            
            return True
            
        except Exception as e:
            warnings.warn(f"Failed to capture image: {e}", RuntimeWarning)
            return False

    def draw_ui(
        self,
        frame: np.ndarray,
        faces: List[Tuple[int, int, int, int]],
        ready: bool,
        message: str,
        quality_score: Optional[float] = None
    ) -> np.ndarray:
        """
        Draw comprehensive user interface overlay on frame.
        
        Args:
            frame: Input frame
            faces: Detected faces
            ready: Whether ready to capture
            message: Status message
            quality_score: Optional quality score
            
        Returns:
            Frame with UI overlay
        """
        display = frame.copy()
        h, w = display.shape[:2]
        
        # Semi-transparent header overlay
        overlay = display.copy()
        header_height = 140
        cv2.rectangle(overlay, (0, 0), (w, header_height), self.COLOR_DARK_BG, -1)
        display = cv2.addWeighted(display, 0.6, overlay, 0.4, 0)
        
        # Progress bar
        progress = self.captured_count / self.num_samples
        bar_width = w - 40
        bar_height = 35
        bar_x, bar_y = 20, 20
        
        # Background
        cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (60, 60, 60), -1)
        cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     self.COLOR_WHITE, 2)
        
        # Progress fill
        fill_width = int(bar_width * progress)
        if fill_width > 0:
            color = self.COLOR_SUCCESS if ready else self.COLOR_INFO
            cv2.rectangle(display, (bar_x + 2, bar_y + 2), 
                         (bar_x + fill_width - 2, bar_y + bar_height - 2),
                         color, -1)
        
        # Progress text
        progress_text = f"{self.captured_count}/{self.num_samples} ({progress*100:.1f}%)"
        text_size = cv2.getTextSize(progress_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = bar_x + (bar_width - text_size[0]) // 2
        cv2.putText(display, progress_text, (text_x, bar_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.COLOR_WHITE, 2)
        
        # Status message with background
        msg_y = 75
        msg_color = self.COLOR_SUCCESS if ready else (
            self.COLOR_WARNING if quality_score is not None else self.COLOR_ERROR
        )
        
        # Message background
        msg_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        msg_bg_x1 = 15
        msg_bg_x2 = msg_size[0] + 35
        cv2.rectangle(display, (msg_bg_x1, msg_y - 25), (msg_bg_x2, msg_y + 5),
                     self.COLOR_DARK_BG, -1)
        
        cv2.putText(display, message, (20, msg_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, msg_color, 2)
        
        # Quality and statistics
        stats_y = 115
        if quality_score is not None:
            quality_text = f"Quality: {quality_score:.2f}"
            quality_color = (
                self.COLOR_SUCCESS if quality_score >= self.EXCELLENT_QUALITY
                else self.COLOR_WARNING if quality_score >= self.min_quality
                else self.COLOR_ERROR
            )
            cv2.putText(display, quality_text, (20, stats_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, quality_color, 2)
        
        # Mode indicator
        mode_text = f"Mode: {self.mode.value.upper()}"
        cv2.putText(display, mode_text, (w - 200, stats_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_WHITE, 2)
        
        # Draw face rectangles with enhanced markers
        for face in faces:
            x, y, fw, fh = face
            rect_color = self.COLOR_SUCCESS if ready else self.COLOR_WARNING
            thickness = 3 if ready else 2
            
            # Main rectangle
            cv2.rectangle(display, (x, y), (x + fw, y + fh), rect_color, thickness)
            
            # Corner markers for better visibility
            corner_len = min(fw, fh) // 4
            corner_thickness = 5
            
            # Top-left
            cv2.line(display, (x, y), (x + corner_len, y), rect_color, corner_thickness)
            cv2.line(display, (x, y), (x, y + corner_len), rect_color, corner_thickness)
            
            # Top-right
            cv2.line(display, (x + fw, y), (x + fw - corner_len, y), rect_color, corner_thickness)
            cv2.line(display, (x + fw, y), (x + fw, y + corner_len), rect_color, corner_thickness)
            
            # Bottom-left
            cv2.line(display, (x, y + fh), (x + corner_len, y + fh), rect_color, corner_thickness)
            cv2.line(display, (x, y + fh), (x, y + fh - corner_len), rect_color, corner_thickness)
            
            # Bottom-right
            cv2.line(display, (x + fw, y + fh), (x + fw - corner_len, y + fh), rect_color, corner_thickness)
            cv2.line(display, (x + fw, y + fh), (x + fw, y + fh - corner_len), rect_color, corner_thickness)
            
            # Face size indicator
            size_text = f"{fw}x{fh}px"
            cv2.putText(display, size_text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, rect_color, 2)
        
        # Footer with keyboard hints
        footer_y = h - 20
        hints = "ESC: Exit | SPACE: Capture | P: Pause | D: Stats"
        cv2.putText(display, hints, (20, footer_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WHITE, 1)
        
        return display

    def flash_effect(self, display: np.ndarray, window_name: str) -> None:
        """Show flash effect after capture."""
        white_frame = np.ones_like(display) * 255
        flash = cv2.addWeighted(display, 0.2, white_frame, 0.8, 0)
        cv2.imshow(window_name, flash)
        cv2.waitKey(self.FLASH_DURATION_MS)
    
    def display_live_stats(self) -> None:
        """Display live statistics in console."""
        elapsed = time.time() - self.session_start_time
        print("\n" + "="*70)
        print("📊 LIVE STATISTICS")
        print("="*70)
        print(f"Captured:      {self.captured_count}/{self.num_samples} ({self.captured_count/self.num_samples*100:.1f}%)")
        print(f"Elapsed time:  {elapsed:.1f}s")
        print(f"Capture rate:  {self.stats.captures_per_minute:.1f} per minute")
        
        if self.stats.quality_scores:
            print(f"Avg quality:   {self.stats.avg_quality_score:.2f}")
            print(f"Quality range: {self.stats.min_quality_score:.2f} - {self.stats.max_quality_score:.2f}")
        
        avg_w, avg_h = self.stats.avg_face_size
        if avg_w > 0:
            print(f"Avg face size: {avg_w:.0f}x{avg_h:.0f} pixels")
        
        print(f"Failures:      Quality={self.stats.quality_failures}, "
              f"Multiple={self.stats.multiple_faces}, "
              f"NoFace={self.stats.no_face_detected}")
        print("="*70 + "\n")

    def run(self) -> CaptureStats:
        """
        Run the capture session.
        
        Returns:
            CaptureStats object with session statistics
        """
        # Initialize camera
        if not self.initialize_camera():
            return self.stats
        
        # Print instructions
        self.print_instructions()
        
        # Start session
        self.status = CaptureStatus.CAPTURING
        self.session_start_time = time.time()
        window_name = f'Training Data Capture - {self.person_name}'
        
        # Create window with specific size
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        try:
            while self.captured_count < self.num_samples:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Error: Failed to read from camera")
                    print("   Camera may have been disconnected")
                    self.status = CaptureStatus.FAILED
                    break
                
                # Skip processing if paused
                if self.status == CaptureStatus.PAUSED:
                    cv2.putText(frame, "PAUSED - Press P to resume", 
                               (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                               1.5, self.COLOR_WARNING, 3)
                    cv2.imshow(window_name, frame)
                    key = cv2.waitKey(30) & 0xFF
                    if key == ord('p') or key == ord('P'):
                        self.status = CaptureStatus.CAPTURING
                        print("▶️  Resumed")
                    elif key == 27 or key == ord('q') or key == ord('Q'):
                        print("\n⚠️  Exiting...")
                        break
                    continue
                
                # Detect faces
                faces = self.detector.detect_faces(frame)
                
                # Assess readiness
                ready, message, face_bbox, quality_score = self.assess_capture_readiness(
                    frame, faces
                )
                
                # Draw UI
                display = self.draw_ui(frame, faces, ready, message, quality_score)
                
                # Handle auto-capture
                if ready and face_bbox is not None and self.should_auto_capture():
                    if self.capture_face(frame, face_bbox, quality_score):
                        self.flash_effect(display, window_name)
                
                # Display frame
                cv2.imshow(window_name, display)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27 or key == ord('q') or key == ord('Q'):  # ESC or Q - exit
                    if self.captured_count > 0:
                        print(f"\n⚠️  Session interrupted by user")
                        response = input(f"You have {self.captured_count} samples. Exit anyway? (y/n): ")
                        if response.lower() == 'y':
                            break
                    else:
                        print(f"\n⚠️  Exiting...")
                        break
                
                elif key == ord('p') or key == ord('P'):  # Pause/Resume
                    self.status = (
                        CaptureStatus.PAUSED if self.status == CaptureStatus.CAPTURING
                        else CaptureStatus.CAPTURING
                    )
                    status_text = "PAUSED" if self.status == CaptureStatus.PAUSED else "RESUMED"
                    print(f"\n⏸️  {status_text}")
                
                elif key == ord('d') or key == ord('D'):  # Display stats
                    self.display_live_stats()
                
                elif key == 32 and self.mode != CaptureMode.AUTO:  # SPACE - manual capture
                    if ready and face_bbox is not None and quality_score is not None:
                        current_time = time.time()
                        if (current_time - self.last_capture_time) > self.MIN_CAPTURE_DELAY:
                            if self.capture_face(frame, face_bbox, quality_score):
                                self.flash_effect(display, window_name)
                        else:
                            print(f"⏱️  Wait {self.MIN_CAPTURE_DELAY:.1f}s between captures")
                    else:
                        print("⚠️  Cannot capture - requirements not met")
                
                self.frame_count += 1
                
        except KeyboardInterrupt:
            print(f"\n⚠️  Capture interrupted by user (Ctrl+C)")
            self.status = CaptureStatus.FAILED
            
        finally:
            # Cleanup
            self.cleanup()
            
            # Calculate final stats
            self.stats.session_duration = time.time() - self.session_start_time
            
            # Set final status
            if self.captured_count >= self.num_samples:
                self.status = CaptureStatus.COMPLETED
            elif self.captured_count > 0:
                self.status = CaptureStatus.PAUSED  # Partially completed
        
        # Print summary
        self.print_summary()
        
        return self.stats

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
    
        # Small delay to ensure windows close
        cv2.waitKey(1)

    def print_summary(self) -> None:
        """Print comprehensive capture session summary."""
        print(f"\n{'='*70}")
        print(f"📊 CAPTURE SESSION SUMMARY")
        print(f"{'='*70}")
        print(f"Person:   {self.person_name}")
        print(f"Status:   {self.status.value.upper()}")
        print(f"\n{self.stats}")
        
        # Quality distribution
        if self.stats.quality_scores:
            quality_bins = {
                'Excellent (≥0.8)': sum(1 for q in self.stats.quality_scores if q >= 0.8),
                'Good (0.6-0.8)': sum(1 for q in self.stats.quality_scores if 0.6 <= q < 0.8),
                'Fair (0.5-0.6)': sum(1 for q in self.stats.quality_scores if 0.5 <= q < 0.6),
                'Poor (<0.5)': sum(1 for q in self.stats.quality_scores if q < 0.5)
            }
            
            print(f"\n📈 Quality Distribution:")
            for category, count in quality_bins.items():
                if count > 0:
                    pct = (count / len(self.stats.quality_scores)) * 100
                    bar = "█" * int(pct / 5)
                    print(f"  {category:20} {count:3} ({pct:5.1f}%) {bar}")
        
        print(f"\n📁 Output Directory: {self.person_dir}")
        print(f"   Processed images: {self.captured_count} files")
        if self.save_original:
            print(f"   Original images:  {self.original_dir}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if self.captured_count < self.num_samples:
            shortage = self.num_samples - self.captured_count
            print(f"  ⚠️  You captured {self.captured_count}/{self.num_samples} samples")
            print(f"     Consider capturing {shortage} more for better accuracy")
            print(f"     Run: python {Path(__file__).name} --name \"{self.person_name}\" --samples {shortage}")
        elif self.captured_count >= self.num_samples:
            print(f"  ✅ Captured sufficient samples ({self.captured_count})")
        
        if self.stats.avg_quality_score < 0.6:
            print(f"  ⚠️  Average quality is low ({self.stats.avg_quality_score:.2f})")
            print(f"     Tips: Better lighting, cleaner lens, move closer")
        elif self.stats.avg_quality_score >= 0.8:
            print(f"  ✅ Excellent average quality ({self.stats.avg_quality_score:.2f})")
        
        if self.stats.success_rate < 50:
            print(f"  ⚠️  Low success rate ({self.stats.success_rate:.1f}%)")
            print(f"     Tips: Improve lighting, reduce motion, one person only")
        
        # Next steps
        print(f"\n🚀 NEXT STEPS:")
        print(f"  1. Review captured images:")
        print(f"     ls {self.person_dir}")
        print(f"\n  2. Capture more people:")
        print(f"     python {Path(__file__).name} --name \"Another Person\" --samples 30")
        print(f"\n  3. Train the model:")
        print(f"     python examples/train_with_validation.py --data {self.output_dir}")
        
        print(f"{'='*70}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Professional training data capture for face recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-capture 30 high-quality samples
  python capture_training_data.py --name "John Doe" --samples 30
  
  # Manual capture with very high quality threshold
  python capture_training_data.py --name "Jane" --mode manual --quality-threshold 0.7
  
  # Hybrid mode with alignment and original saving
  python capture_training_data.py --name "Bob" --mode hybrid --enable-alignment --save-original
  
  # Use different camera
  python capture_training_data.py --name "Alice" --camera 1

Quality Guidelines:
  • 0.5-0.6: Acceptable (minimum for training)
  • 0.6-0.7: Good quality (recommended default)
  • 0.7-0.8: Very good quality
  • 0.8+:    Excellent quality (strict)

Sample Guidelines:
  • Minimum:     10 samples (poor accuracy)
  • Recommended: 20-30 samples (good accuracy)
  • Optimal:     50+ samples (excellent accuracy)
        """
    )
    
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name of the person to capture"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=30,
        help="Number of samples to capture (default: 30)"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="training_data",
        help="Output directory for training data (default: training_data)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "manual", "hybrid"],
        default="auto",
        help="Capture mode (default: auto)"
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.6,
        help="Minimum quality score 0.0-1.0 (default: 0.6)"
    )
    parser.add_argument(
        "--no-clahe",
        action="store_true",
        help="Disable CLAHE preprocessing"
    )
    parser.add_argument(
        "--enable-alignment",
        action="store_true",
        help="Enable face alignment (slower but higher quality)"
    )
    parser.add_argument(
        "--save-original",
        action="store_true",
        help="Save original unprocessed images alongside processed"
    )
    parser.add_argument(
        "--no-diversity-check",
        action="store_true",
        help="Disable pose diversity checking"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.samples < 5:
        print("❌ Error: Minimum 5 samples required")
        sys.exit(1)
    
    if args.samples < 10:
        print("⚠️  Warning: Less than 10 samples may result in poor recognition accuracy")
        print("   Recommended: 20-50 samples per person\n")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    if not 0.0 <= args.quality_threshold <= 1.0:
        print(f"❌ Error: quality-threshold must be between 0.0 and 1.0")
        sys.exit(1)
    
    if args.quality_threshold < 0.5:
        print("⚠️  Warning: Very low quality threshold may capture poor images")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # Create and run capture session
    capture_mode = CaptureMode(args.mode)
    
    print("\n" + "="*70)
    print("🎥 FACE RECOGNITION TRAINING DATA CAPTURE")
    print("="*70)
    print(f"Version: 2.0 (High Quality)")
    print(f"Resolution: 1920x1080 (1080p)")
    print(f"Face Resolution: 150x150 pixels")
    print(f"Min Face Size: 100x100 pixels")
    print("="*70 + "\n")
    
    capturer = TrainingDataCapture(
        person_name=args.name,
        num_samples=args.samples,
        camera_index=args.camera,
        output_dir=args.output,
        mode=capture_mode,
        min_quality=args.quality_threshold,
        use_clahe=not args.no_clahe,
        enable_alignment=args.enable_alignment,
        save_original=args.save_original,
        enable_diversity_check=not args.no_diversity_check
    )
    
    stats = capturer.run()
    
    # Exit with appropriate code
    if stats.total_captured >= args.samples:
        print("✅ Session completed successfully!")
        sys.exit(0)  # Success
    elif stats.total_captured >= args.samples * 0.7:
        print("⚠️  Session partially completed (70%+ captured)")
        sys.exit(1)  # Partial success
    else:
        print("❌ Session incomplete (less than 70% captured)")
        sys.exit(2)  # Failure


if __name__ == "__main__":
    main()