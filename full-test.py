"""
Live Test - Anti-Spoofing + Face Recognition
===========================================

Real-time webcam testing for integrated anti-spoofing and face recognition.

Features:
- Real-time anti-spoofing detection
- Face recognition with liveness verification
- Visual feedback with confidence scores
- Performance metrics display
- Multiple security levels
- Keyboard controls
- FPS counter

Usage:
    python live_test_antispoofing_recognition.py [--model MODEL_PATH] [--level LEVEL]

Controls:
    ESC/Q - Quit
    1 - Lenient mode
    2 - Balanced mode (default)
    3 - Strict mode
    4 - Paranoid mode
    D - Toggle debug info
    S - Toggle statistics
    F - Toggle face detection
    R - Reset statistics
    SPACE - Take screenshot
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

# Import our modules (adjust paths as needed)
try:
    from src.face.recognizer import FaceRecognizer
    from src.face.final_anti_spoof import (
        UltimateAntiSpoof,
        SecurityLevel,
        AttackType,
        AntiSpoofResult
    )
    from generate_report import TestReportGenerator
except ImportError as e:
    print(f"ERROR: Could not import required modules: {e}")
    print("Make sure you're running from the project root directory.")
    sys.exit(1)


# ============================================================================
# COLORS AND CONSTANTS
# ============================================================================

# Colors (BGR format)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_ORANGE = (0, 165, 255)
COLOR_CYAN = (255, 255, 0)

# Display settings
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1
LINE_HEIGHT = 20


# ============================================================================
# PERFORMANCE TRACKER
# ============================================================================

class PerformanceTracker:
    """Track performance metrics for the live system."""
    
    def __init__(self):
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.fps = 0.0
        
        # Anti-spoofing stats
        self.total_checks = 0
        self.real_detections = 0
        self.spoof_detections = 0
        
        # Recognition stats
        self.recognition_attempts = 0
        self.successful_recognitions = 0
        
        # Timing
        self.avg_processing_time = 0.0
        self.processing_times = []
        
    def update_fps(self):
        """Update FPS calculation."""
        self.frame_count += 1
        elapsed = time.time() - self.fps_start_time
        if elapsed > 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_start_time = time.time()
    
    def add_antispoofing_result(self, is_real: bool):
        """Add anti-spoofing result."""
        self.total_checks += 1
        if is_real:
            self.real_detections += 1
        else:
            self.spoof_detections += 1
    
    def add_recognition_result(self, recognized: bool):
        """Add recognition result."""
        self.recognition_attempts += 1
        if recognized:
            self.successful_recognitions += 1
    
    def add_processing_time(self, processing_time: float):
        """Add processing time."""
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 30:
            self.processing_times.pop(0)
        self.avg_processing_time = np.mean(self.processing_times)
    
    def reset(self):
        """Reset all statistics."""
        self.total_checks = 0
        self.real_detections = 0
        self.spoof_detections = 0
        self.recognition_attempts = 0
        self.successful_recognitions = 0
        self.processing_times = []
        self.avg_processing_time = 0.0
    
    def get_stats(self) -> Dict:
        """Get all statistics."""
        return {
            'fps': self.fps,
            'total_checks': self.total_checks,
            'real_rate': f"{self.real_detections / max(self.total_checks, 1) * 100:.1f}%",
            'spoof_rate': f"{self.spoof_detections / max(self.total_checks, 1) * 100:.1f}%",
            'recognition_rate': f"{self.successful_recognitions / max(self.recognition_attempts, 1) * 100:.1f}%",
            'avg_time': self.avg_processing_time
        }


# ============================================================================
# VISUAL OVERLAY
# ============================================================================

class VisualOverlay:
    """Handle all visual overlays on the video feed."""
    
    @staticmethod
    def draw_header(frame: np.ndarray, title: str, mode: str):
        """Draw header with title and mode."""
        h, w = frame.shape[:2]
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), COLOR_BLACK, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Title
        cv2.putText(frame, title, (10, 25), FONT, 0.7, COLOR_CYAN, 2)
        
        # Mode
        mode_text = f"Mode: {mode.upper()}"
        cv2.putText(frame, mode_text, (10, 45), FONT, 0.5, COLOR_YELLOW, 1)
    
    @staticmethod
    def draw_face_box(
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        is_real: bool,
        label: str,
        confidence: float
    ):
        """Draw bounding box around face with labels."""
        x, y, w, h = bbox
        
        # Choose color based on result
        color = COLOR_GREEN if is_real else COLOR_RED
        
        # Draw box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Draw label background
        label_text = f"{label} ({confidence:.1%})"
        (text_w, text_h), _ = cv2.getTextSize(label_text, FONT, FONT_SCALE, FONT_THICKNESS)
        cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w + 10, y), color, -1)
        
        # Draw label text
        cv2.putText(frame, label_text, (x + 5, y - 5), FONT, FONT_SCALE, COLOR_WHITE, FONT_THICKNESS)
    
    @staticmethod
    def draw_status_panel(
        frame: np.ndarray,
        is_real: bool,
        attack_type: str,
        texture: float,
        motion: float,
        color: float,
        depth: float,
        frequency: float,
        color_temp: float = 0.5,
        refresh: float = 0.5,
        rppg: float = 0.5
    ):
        """Draw anti-spoofing status panel."""
        h, w = frame.shape[:2]
        panel_x = w - 280
        panel_y = 60
        
        # Semi-transparent background (taller to fit new metrics)
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (w - 10, panel_y + 230), COLOR_BLACK, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Status
        status_text = "[OK] REAL" if is_real else "[ERROR] SPOOF"
        status_color = COLOR_GREEN if is_real else COLOR_RED
        cv2.putText(frame, status_text, (panel_x + 10, panel_y + 25), 
                   FONT, 0.6, status_color, 2)
        
        if not is_real:
            cv2.putText(frame, attack_type, (panel_x + 10, panel_y + 45),
                       FONT, 0.4, COLOR_ORANGE, 1)
        
        # Individual scores (now 8 total: 5 original + 3 new)
        y_offset = panel_y + 70
        scores = [
            ("Texture", texture),
            ("Motion", motion),
            ("Color", color),
            ("Depth", depth),
            ("Frequency", frequency),
            ("ColorTemp", color_temp),  # NEW
            ("Refresh", refresh),       # NEW
            ("rPPG", rppg)              # NEW
        ]
        
        for label, score in scores:
            VisualOverlay._draw_score_bar(frame, panel_x + 10, y_offset, label, score)
            y_offset += 20
    
    @staticmethod
    def _draw_score_bar(frame: np.ndarray, x: int, y: int, label: str, score: float):
        """Draw a score bar."""
        bar_width = 120
        bar_height = 12
        
        # Label
        cv2.putText(frame, f"{label}:", (x, y + 10), FONT, 0.35, COLOR_WHITE, 1)
        
        # Bar background
        cv2.rectangle(frame, (x + 70, y), (x + 70 + bar_width, y + bar_height), 
                     COLOR_WHITE, 1)
        
        # Fill bar
        fill_width = int(bar_width * score)
        color = COLOR_GREEN if score >= 0.5 else COLOR_RED
        if fill_width > 0:
            cv2.rectangle(frame, (x + 70, y), (x + 70 + fill_width, y + bar_height),
                         color, -1)
        
        # Score value
        cv2.putText(frame, f"{score:.2f}", (x + 195, y + 10), 
                   FONT, 0.35, COLOR_WHITE, 1)
    
    @staticmethod
    def draw_recognition_panel(
        frame: np.ndarray,
        recognized: bool,
        name: str,
        confidence: float,
        quality: float
    ):
        """Draw recognition status panel."""
        h, w = frame.shape[:2]
        panel_x = 10
        panel_y = 60
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + 250, panel_y + 100), 
                     COLOR_BLACK, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Recognition status
        if recognized:
            status = f"[OK] {name}"
            color = COLOR_GREEN
        else:
            status = f"[ERROR] {name}"
            color = COLOR_RED
        
        cv2.putText(frame, status, (panel_x + 10, panel_y + 25), 
                   FONT, 0.6, color, 2)
        
        # Confidence
        conf_text = f"Confidence: {confidence:.1f}"
        cv2.putText(frame, conf_text, (panel_x + 10, panel_y + 50),
                   FONT, 0.45, COLOR_WHITE, 1)
        
        # Quality
        quality_text = f"Quality: {quality:.1%}"
        quality_color = COLOR_GREEN if quality > 0.7 else COLOR_YELLOW if quality > 0.5 else COLOR_RED
        cv2.putText(frame, quality_text, (panel_x + 10, panel_y + 75),
                   FONT, 0.45, quality_color, 1)
    
    @staticmethod
    def draw_statistics(frame: np.ndarray, stats: Dict):
        """Draw performance statistics."""
        h, w = frame.shape[:2]
        panel_x = 10
        panel_y = h - 150
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + 220, h - 10),
                     COLOR_BLACK, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "Statistics", (panel_x + 10, panel_y + 20),
                   FONT, 0.5, COLOR_CYAN, 1)
        
        # Stats
        y_offset = panel_y + 40
        stat_items = [
            ("FPS", f"{stats['fps']:.1f}"),
            ("Checks", str(stats['total_checks'])),
            ("Real Rate", stats['real_rate']),
            ("Recognition", stats['recognition_rate']),
            ("Avg Time", f"{stats['avg_time']:.1f}ms")
        ]
        
        for label, value in stat_items:
            cv2.putText(frame, f"{label}: {value}", (panel_x + 10, y_offset),
                       FONT, 0.4, COLOR_WHITE, 1)
            y_offset += 20
    
    @staticmethod
    def draw_controls(frame: np.ndarray):
        """Draw control hints."""
        h, w = frame.shape[:2]
        panel_x = w - 260
        panel_y = h - 135  # Increased from 120 to fit new control
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (w - 10, h - 10),
                     COLOR_BLACK, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "Controls", (panel_x + 10, panel_y + 20),
                   FONT, 0.5, COLOR_CYAN, 1)
        
        # Controls
        controls = [
            "1-4: Security levels",
            "D: Toggle debug",
            "S: Toggle stats",
            "G: Generate report",
            "R: Reset stats",
            "Q/ESC: Quit"
        ]
        
        y_offset = panel_y + 35
        for control in controls:
            cv2.putText(frame, control, (panel_x + 10, y_offset),
                       FONT, 0.35, COLOR_WHITE, 1)
            y_offset += 15


# ============================================================================
# MAIN LIVE TEST CLASS
# ============================================================================

class LiveTestSystem:
    """Main live test system integrating anti-spoofing and recognition."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        initial_level: str = "balanced",
        camera_id: int = 0
    ):
        """
        Initialize the live test system.
        
        Args:
            model_path: Path to trained face recognition model (optional)
            initial_level: Initial security level
            camera_id: Camera device ID
        """
        print("=" * 70)
        print("Live Test System - Anti-Spoofing + Face Recognition")
        print("=" * 70)
        
        # Initialize anti-spoofing
        print("\n[1/4] Initializing anti-spoofing detector...")
        # Map security levels to SecurityLevel enum
        level_mapping = {
            'lenient': SecurityLevel.LENIENT,
            'balanced': SecurityLevel.BALANCED,
            'strict': SecurityLevel.STRICT,
            'paranoid': SecurityLevel.PARANOID
        }
        self.security_level = level_mapping.get(initial_level, SecurityLevel.BALANCED)
        self.current_level = initial_level
        
        # Initialize anti-spoofing detector with new UltimateAntiSpoof system
        self.antispoofing = UltimateAntiSpoof(
            level=self.security_level.value,  # Convert enum to string value
            enable_video_mode=True,
            debug=False
        )
        
        # Initialize face recognition
        print("[2/4] Initializing face recognizer...")
        self.recognizer = None
        self.recognition_enabled = False
        
        if model_path and Path(model_path).exists():
            try:
                print("  Creating FaceRecognizer...")
                import cv2
                print(f"    OpenCV version: {cv2.__version__}")
                print(f"    Face module available: {hasattr(cv2, 'face')}")
                if hasattr(cv2, 'face'):
                    print(f"    LBPH available: {hasattr(cv2.face, 'LBPHFaceRecognizer_create')}")
                
                self.recognizer = FaceRecognizer(
                    threshold=50.0,
                    enable_antispoofing=True,  # Enable integrated anti-spoofing
                    antispoofing_mode="basic",  # Use basic mode (balanced)
                    reject_on_spoof=True  # Reject spoofed faces
                )
                print("  FaceRecognizer created successfully")
                print("  Loading model...")
                self.recognizer.load_model(model_path)
                self.recognition_enabled = True
                print(f"  [OK] Loaded model with {len(self.recognizer.get_known_people())} people")
                print(f"  Known people: {', '.join(self.recognizer.get_known_people())}")
            except Exception as e:
                print(f"  [ERROR] Failed to load model: {e}")
                print("  Continuing with anti-spoofing only...")
                self.recognizer = None
                self.recognition_enabled = False
        else:
            print("  [INFO] No model provided - anti-spoofing only mode")
        
        # Initialize face detector
        print("[3/4] Initializing face detector...")
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Initialize camera
        print("[4/4] Initializing camera...")
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")
        
        # Set camera properties for better quality
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  [OK] Camera resolution: {actual_width}x{actual_height}")
        
        # Initialize performance tracker and overlay
        self.perf_tracker = PerformanceTracker()
        self.overlay = VisualOverlay()
        
        # Initialize report generator
        self.report_generator = TestReportGenerator()
        self.enable_reporting = True
        
        # Display settings
        self.show_statistics = True
        self.show_debug = True
        self.show_controls = True
        
        # Last results (for display)
        self.last_antispoofing_result = None
        self.last_recognition_result = None
        self.last_face_bbox = None
        
        print("\n[OK] System ready!")
        print("\nStarting live test... Press Q or ESC to quit.")
        print("Press 'G' to generate report at any time.")
        print("=" * 70 + "\n")
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Processed frame with overlays
        """
        start_time = time.time()
        
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )
        
        # Process each face
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            # Anti-spoofing check
            antispoofing_result = self.antispoofing.check_video_frame(face_roi)
            self.last_antispoofing_result = antispoofing_result
            self.last_face_bbox = (x, y, w, h)
            
            # Track anti-spoofing result
            self.perf_tracker.add_antispoofing_result(antispoofing_result.is_real)
            
            # Record to report generator
            if self.enable_reporting:
                recognition_name = None
                recognition_conf = None
                
                # Recognition (only if real and enabled)
                if self.recognition_enabled and antispoofing_result.is_real and self.recognizer:
                    try:
                        recognition_result = self.recognizer.predict_with_name(face_roi)
                        self.last_recognition_result = recognition_result
                        self.perf_tracker.add_recognition_result(recognition_result.recognized)
                        if recognition_result.recognized:
                            recognition_name = recognition_result.name
                            recognition_conf = recognition_result.confidence
                    except Exception as e:
                        print(f"Recognition error: {e}")
                        self.last_recognition_result = None
                else:
                    self.last_recognition_result = None
                
                # Record frame data to report
                self.report_generator.record_frame(
                    is_real=antispoofing_result.is_real,
                    confidence=antispoofing_result.confidence,
                    attack_type=antispoofing_result.attack_type.value,
                    metrics={
                        'texture_score': antispoofing_result.metrics.texture_score,
                        'motion_score': antispoofing_result.metrics.motion_score,
                        'color_score': antispoofing_result.metrics.color_score,
                        'depth_score': antispoofing_result.metrics.depth_score,
                        'frequency_score': antispoofing_result.metrics.frequency_score,
                        'blink_score': antispoofing_result.metrics.blink_score,
                        'pulse_score': antispoofing_result.metrics.pulse_score,
                        'color_temp_score': antispoofing_result.metrics.color_temp_score,
                        'refresh_score': antispoofing_result.metrics.refresh_score,
                        'rppg_score': antispoofing_result.metrics.rppg_score,
                    },
                    warnings=antispoofing_result.warnings,
                    recognition_name=recognition_name,
                    recognition_confidence=recognition_conf,
                    fps=self.perf_tracker.fps,
                    processing_time=0  # Will be calculated below
                )
            
            # Draw face box (only if we have results)
            if self.last_antispoofing_result:
                antispoofing_result = self.last_antispoofing_result
                if antispoofing_result.is_real:
                    if self.last_recognition_result and self.last_recognition_result.recognized:
                        label = self.last_recognition_result.name
                        confidence = 1.0 - (self.last_recognition_result.confidence / 100.0)
                    else:
                        label = "Real (Unknown)"
                        confidence = antispoofing_result.confidence
                else:
                    label = antispoofing_result.attack_type.value
                    confidence = 1.0 - antispoofing_result.confidence
                
                self.overlay.draw_face_box(
                    frame, (x, y, w, h),
                    antispoofing_result.is_real,
                    label,
                    confidence
                )
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        self.perf_tracker.add_processing_time(processing_time)
        
        return frame
    
    def draw_overlays(self, frame: np.ndarray) -> np.ndarray:
        """Draw all overlays on the frame."""
        # Header
        mode_text = f"{self.current_level} {'+ Recognition' if self.recognition_enabled else ''}"
        self.overlay.draw_header(frame, "Live Anti-Spoofing Test", mode_text)
        
        # Anti-spoofing panel
        if self.show_debug and self.last_antispoofing_result:
            # Extract metrics from the new UltimateAntiSpoof result structure
            metrics = self.last_antispoofing_result.metrics
            self.overlay.draw_status_panel(
                frame,
                self.last_antispoofing_result.is_real,
                self.last_antispoofing_result.attack_type.value,
                metrics.texture_score,
                metrics.motion_score,
                metrics.color_score,
                metrics.depth_score,
                metrics.frequency_score,
                metrics.color_temp_score,  # NEW
                metrics.refresh_score,     # NEW
                metrics.rppg_score         # NEW
            )
        
        # Recognition panel - always show when recognition is enabled
        if self.recognition_enabled:
            if self.last_recognition_result:
                # Show actual recognition result
                self.overlay.draw_recognition_panel(
                    frame,
                    self.last_recognition_result.recognized,
                    self.last_recognition_result.name,
                    self.last_recognition_result.confidence,
                    self.last_recognition_result.quality_score
                )
            else:
                # Show "No face detected" or "Waiting for face"
                self.overlay.draw_recognition_panel(
                    frame,
                    False,  # Not recognized
                    "No Face Detected",
                    0.0,    # No confidence
                    0.0     # No quality
                )
        
        # Statistics
        if self.show_statistics:
            stats = self.perf_tracker.get_stats()
            self.overlay.draw_statistics(frame, stats)
        
        # Controls
        if self.show_controls:
            self.overlay.draw_controls(frame)
        
        return frame
    
    def change_security_level(self, level: str):
        """Change security level by recreating the anti-spoofing detector."""
        self.current_level = level
        # Map security levels to SecurityLevel enum
        level_mapping = {
            'lenient': SecurityLevel.LENIENT,
            'balanced': SecurityLevel.BALANCED,
            'strict': SecurityLevel.STRICT,
            'paranoid': SecurityLevel.PARANOID
        }
        new_level = level_mapping.get(level, SecurityLevel.BALANCED)
        self.security_level = new_level
        
        # Recreate anti-spoofing detector with new level
        if self.antispoofing:
            self.antispoofing = UltimateAntiSpoof(
                level=new_level.value,  # Convert enum to string
                enable_video_mode=True,
                debug=False
            )
        
        print(f"[OK] Security level changed to: {level.upper()}")
    
    def run(self):
        """Run the main loop."""
        try:
            while True:
                # Update FPS
                self.perf_tracker.update_fps()
                
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break
                
                # Process frame
                frame = self.process_frame(frame)
                
                # Draw overlays
                frame = self.draw_overlays(frame)
                
                # Display
                cv2.imshow('Live Test - Anti-Spoofing + Recognition', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # Q or ESC
                    break
                elif key == ord('1'):
                    self.change_security_level('lenient')
                elif key == ord('2'):
                    self.change_security_level('balanced')
                elif key == ord('3'):
                    self.change_security_level('strict')
                elif key == ord('4'):
                    self.change_security_level('paranoid')
                elif key == ord('d'):
                    self.show_debug = not self.show_debug
                    print(f"Debug info: {'ON' if self.show_debug else 'OFF'}")
                elif key == ord('s'):
                    self.show_statistics = not self.show_statistics
                    print(f"Statistics: {'ON' if self.show_statistics else 'OFF'}")
                elif key == ord('c'):
                    self.show_controls = not self.show_controls
                    print(f"Controls: {'ON' if self.show_controls else 'OFF'}")
                elif key == ord('r'):
                    self.perf_tracker.reset()
                    print("[OK] Statistics reset")
                elif key == ord('g'):
                    # Generate report
                    print("\n[*] Generating test report...")
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    report_name = f"live_test_{timestamp}"
                    self.report_generator.generate_report(
                        report_name=report_name,
                        include_visualizations=True
                    )
                elif key == ord(' '):
                    # Take screenshot
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    # Create screenshots directory if it doesn't exist
                    screenshots_dir = Path("screenshots")
                    screenshots_dir.mkdir(exist_ok=True)
                    filename = screenshots_dir / f"screenshot_{timestamp}.jpg"
                    cv2.imwrite(str(filename), frame)
                    print(f"[OK] Screenshot saved: {filename}")
                    print(f"[OK] Screenshot saved: {filename}")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("\nCleaning up...")
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        print("\n" + "=" * 70)
        print("FINAL STATISTICS")
        print("=" * 70)
        stats = self.perf_tracker.get_stats()
        print(f"Total checks: {stats['total_checks']}")
        print(f"Real detection rate: {stats['real_rate']}")
        print(f"Spoof detection rate: {stats['spoof_rate']}")
        if self.recognition_enabled:
            print(f"Recognition rate: {stats['recognition_rate']}")
        print(f"Average processing time: {stats['avg_time']:.1f}ms")
        print(f"Average FPS: {stats['fps']:.1f}")
        print("=" * 70)
        
        # Auto-generate final report
        if self.enable_reporting and self.report_generator.session_data['total_frames'] > 0:
            print("\n[*] Auto-generating final test report...")
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            report_name = f"session_{timestamp}"
            try:
                self.report_generator.generate_report(
                    report_name=report_name,
                    include_visualizations=True
                )
            except Exception as e:
                print(f"[!] Report generation failed: {e}")
                print("    (This is non-critical, session data can be reviewed from stats above)")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Live Test - Anti-Spoofing + Face Recognition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Anti-spoofing only
  python live_test_antispoofing_recognition.py
  
  # With face recognition
  python live_test_antispoofing_recognition.py --model models/face_recognizer.yml
  
  # With specific security level
  python live_test_antispoofing_recognition.py --model models/face_recognizer.yml --level strict
  
  # Use different camera
  python live_test_antispoofing_recognition.py --camera 1

Controls:
  ESC/Q - Quit
  1-4   - Security levels (lenient/balanced/strict/paranoid)
  D     - Toggle debug info
  S     - Toggle statistics
  C     - Toggle controls display
  R     - Reset statistics
  SPACE - Take screenshot
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='Path to trained face recognition model (optional)'
    )
    
    parser.add_argument(
        '--level', '-l',
        type=str,
        default='balanced',
        choices=['lenient', 'balanced', 'strict', 'paranoid'],
        help='Initial security level (default: balanced)'
    )
    
    parser.add_argument(
        '--camera', '-c',
        type=int,
        default=0,
        help='Camera device ID (default: 0)'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize system
        system = LiveTestSystem(
            model_path=args.model,
            initial_level=args.level,
            camera_id=args.camera
        )
        
        # Run
        system.run()
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())