"""
PERFECT ATTENDANCE MANAGEMENT SYSTEM v2.0 - PRODUCTION READY
============================================================

Production-ready attendance system with:
✅ Real-time face recognition with anti-spoofing
✅ Automated attendance logging with timestamps
✅ SQLite database + CSV backup
✅ Per-user detailed reports and analytics
✅ Quality-aware recognition with confidence scoring
✅ Export to Excel/CSV/JSON formats
✅ Dashboard and statistics
✅ Duplicate prevention and session management
✅ Comprehensive error handling
✅ Memory management and cleanup
✅ Thread-safe database operations
✅ Graceful shutdown handling
✅ Logging system
✅ Multi-security level support

Usage:
    python attendance_system.py --mode live
    python attendance_system.py --mode stats --date 2025-11-03
    python attendance_system.py --mode export --format excel
    python attendance_system.py --mode user --name "John Doe" --days 7
    python attendance_system.py --mode backup

Author: OmarBadrawyyy
Version: 2.0.0
"""

import atexit
import csv
import json
import logging
import shutil
import signal
import sqlite3
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Fix imports - handle multiple run locations
try:
    # Get the project root (parent of attendance_system folder)
    current_file = Path(__file__).resolve()
    attendance_system_dir = current_file.parent
    project_root = attendance_system_dir.parent
    
    # Add project root to sys.path so we can import from src
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Now try to import from src.face
    from src.face.detector import FaceDetector
    from src.face.recognizer import FaceRecognizer
    from src.face.final_anti_spoof import UltimateAntiSpoof, SecurityLevel
    from src.face.per_user_report_generator import PerUserReportGenerator
    
    print(f"✓ Imports successful from: {project_root}")
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("\n" + "="*70)
    print("IMPORT ERROR - Required modules not found")
    print("="*70)
    print("Required files:")
    print("  - src/face/detector.py")
    print("  - src/face/recognizer.py")
    print("  - src/face/final_anti_spoof.py")
    print("  - src/face/per_user_report_generator.py")
    print("\nPlease ensure:")
    print("  1. All required files exist in src/face/")
    print("  2. The src/face/ directory has __init__.py")
    
    # Safe project root display
    try:
        current_file_debug = Path(__file__).resolve()
        project_root_debug = current_file_debug.parent.parent
        print(f"\nProject root detected as: {project_root_debug}")
        print(f"Looking for: {str(project_root_debug / 'src' / 'face')}")
    except Exception:
        pass
    
    print("="*70)
    sys.exit(1)


# Setup logging
def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"attendance_{datetime.now().strftime('%Y%m%d')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


class AttendanceRecord:
    """Single attendance record with comprehensive validation."""
    
    def __init__(
        self,
        user_id: int,
        name: str,
        timestamp: datetime,
        confidence: float,
        quality: float,
        is_live: bool,
        check_type: str = "IN"
    ):
        """
        Initialize attendance record with validation.
        
        Args:
            user_id: User's unique identifier
            name: User's name
            timestamp: Timestamp of the check
            confidence: Recognition confidence score
            quality: Face image quality score
            is_live: Liveness detection result
            check_type: "IN" or "OUT"
            
        Raises:
            ValueError: If validation fails
        """
        # Validate check_type
        if check_type not in ["IN", "OUT"]:
            raise ValueError(f"check_type must be 'IN' or 'OUT', got: {check_type}")
        
        # Validate name
        if not name or not isinstance(name, str):
            raise ValueError("name must be a non-empty string")
        
        # Validate user_id
        if not isinstance(user_id, int) or user_id < 0:
            raise ValueError("user_id must be a non-negative integer")
        
        # Validate timestamp
        if not isinstance(timestamp, datetime):
            raise ValueError("timestamp must be a datetime object")
        
        self.user_id = user_id
        self.name = name
        self.timestamp = timestamp
        self.confidence = float(confidence)
        self.quality = float(quality)
        self.is_live = bool(is_live)
        self.check_type = check_type
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'user_id': self.user_id,
            'name': self.name,
            'timestamp': self.timestamp.isoformat(),
            'date': self.timestamp.strftime('%Y-%m-%d'),
            'time': self.timestamp.strftime('%H:%M:%S'),
            'confidence': round(self.confidence, 2),
            'quality': round(self.quality, 3),
            'is_live': self.is_live,
            'check_type': self.check_type
        }
    
    def __str__(self) -> str:
        """Human-readable representation."""
        return (f"{self.check_type}: {self.name} at {self.timestamp.strftime('%H:%M:%S')} "
                f"(conf: {self.confidence:.1f}, quality: {self.quality:.2f})")
    
    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (f"AttendanceRecord(user_id={self.user_id}, name='{self.name}', "
                f"timestamp={self.timestamp.isoformat()}, check_type='{self.check_type}')")


class AttendanceManager:
    """
    Core attendance management system with comprehensive features.
    
    Features:
    - SQLite database with transaction safety + CSV backup
    - Thread-safe database operations
    - Duplicate prevention with configurable threshold
    - Automatic memory management and cleanup
    - Comprehensive error handling
    - Multi-format export (Excel/CSV/JSON)
    - Real-time statistics and reporting
    - Graceful shutdown handling
    - Logging system
    """
    
    # Constants
    MAX_RECORDS_IN_MEMORY = 1000
    CLEANUP_INTERVAL_HOURS = 1
    DATABASE_VERSION = 1
    
    def __init__(
        self,
        model_path: str = "models/combined_model.yml",
        data_dir: str = "attendance_data",
        security_level: str = "balanced",
        duplicate_threshold_minutes: int = 5,
        enable_antispoofing: bool = True,
        enable_reports: bool = True
    ):
        """
        Initialize attendance management system.
        
        Args:
            model_path: Path to trained recognition model
            data_dir: Directory for attendance data storage
            security_level: Security level (lenient/balanced/strict)
            duplicate_threshold_minutes: Minutes to prevent duplicate check-ins
            enable_antispoofing: Enable anti-spoofing detection
            enable_reports: Enable per-user reporting
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        for subdir in ['logs', 'reports', 'exports', 'backups']:
            (self.data_dir / subdir).mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = setup_logging(self.data_dir / "logs")
        
        self.model_path = model_path
        self.security_level = security_level
        self.duplicate_threshold = timedelta(minutes=duplicate_threshold_minutes)
        
        # Thread safety
        self._db_lock = threading.Lock()
        self._shutdown_requested = False
        
        # Register cleanup handlers
        atexit.register(self._cleanup_on_exit)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        self.logger.info("="*70)
        self.logger.info("🚀 Initializing Attendance Management System v2.0")
        self.logger.info("="*70)
        self.logger.info(f"   Security Level: {security_level.upper()}")
        self.logger.info(f"   Data Directory: {self.data_dir}")
        self.logger.info(f"   Model Path: {model_path}")
        
        try:
            # Face detector
            self.detector = FaceDetector(
                use_clahe=True,
                min_neighbors=5,
                scale_factor=1.1
            )
            self.logger.info("   ✅ Face detector initialized")
            
            # Face recognizer
            self.recognizer = FaceRecognizer(
                threshold=self._get_threshold_for_level(security_level),
                use_enhanced_preprocessing=True,
                enable_antispoofing=enable_antispoofing,
                antispoofing_mode="basic" if security_level != "strict" else "high_security",
                reject_on_spoof=True
            )
            
            # Load model if exists
            if Path(model_path).exists():
                self.recognizer.load_model(model_path)
                known_people = self.recognizer.get_known_people()
                self.logger.info(f"   ✅ Model loaded: {len(known_people)} people")
                
                if known_people:
                    people_preview = ', '.join(known_people[:5])
                    if len(known_people) > 5:
                        people_preview += f" ... (+{len(known_people) - 5} more)"
                    self.logger.info(f"      People: {people_preview}")
            else:
                self.logger.warning(f"   ⚠️  No model found at: {model_path}")
                self.logger.warning("      Please train the system first using train_model.py")
            
            # Anti-spoofing detector (optional)
            self.antispoofing = None
            if enable_antispoofing:
                try:
                    self.antispoofing = UltimateAntiSpoof(
                        level=security_level,
                        enable_video_mode=True
                    )
                    self.logger.info("   ✅ Anti-spoofing enabled")
                except Exception as e:
                    self.logger.warning(f"   ⚠️  Anti-spoofing initialization failed: {e}")
            
            # Report generator (optional)
            self.report_generator = None
            if enable_reports:
                try:
                    self.report_generator = PerUserReportGenerator(
                        db_path=str(self.data_dir / "user_reports.db"),
                        reports_dir=str(self.data_dir / "reports")
                    )
                    self.logger.info("   ✅ Report generator enabled")
                except Exception as e:
                    self.logger.warning(f"   ⚠️  Report generator initialization failed: {e}")
            
        except Exception as e:
            self.logger.error(f"   ❌ Initialization error: {e}")
            raise
        
        # Initialize database
        self.db_path = self.data_dir / "attendance.db"
        self._init_database()
        
        # Session tracking
        self.attendance_records: List[AttendanceRecord] = []
        self.last_check_in: Dict[str, datetime] = {}
        self.last_check_out: Dict[str, datetime] = {}
        self.daily_stats: Dict[str, Dict] = defaultdict(lambda: {
            'check_ins': 0,
            'check_outs': 0,
            'total_time': timedelta(),
            'last_check_in_time': None
        })
        
        # Memory management
        self._last_cleanup = datetime.now()
        self._cleanup_interval = timedelta(hours=self.CLEANUP_INTERVAL_HOURS)
        
        # Load today's records
        self._load_today_records()
        
        self.logger.info("✅ System ready!")
        self.logger.info("="*70 + "\n")
    
    def _signal_handler(self, sig: int, frame) -> None:
        """Handle Ctrl+C gracefully."""
        self.logger.warning("\n⚠️  Shutdown signal received (Ctrl+C)")
        self._shutdown_requested = True
        self._cleanup_on_exit()
        sys.exit(0)
    
    def _cleanup_on_exit(self) -> None:
        """Cleanup when exiting - called automatically."""
        if self._shutdown_requested:
            return  # Already cleaned up
        
        self._shutdown_requested = True
        
        try:
            self.logger.info("🧹 Performing cleanup...")
            
            # Generate final reports
            if self.report_generator:
                try:
                    self.report_generator.finalize_session()
                    self.logger.info("   ✅ Reports generated")
                except Exception as e:
                    self.logger.warning(f"   ⚠️  Report generation failed: {e}")
            
            # Create auto-backup
            try:
                backup_name = f"auto_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
                self.backup_database(backup_name)
                self.logger.info("   ✅ Database backed up")
            except Exception as e:
                self.logger.warning(f"   ⚠️  Backup failed: {e}")
            
            self.logger.info("✅ Cleanup complete")
            
        except Exception as e:
            self.logger.error(f"⚠️  Cleanup error: {e}")
    
    def _get_threshold_for_level(self, level: str) -> float:
        """Get recognition threshold for security level."""
        thresholds = {
            'lenient': 60.0,
            'balanced': 50.0,
            'strict': 40.0
        }
        return thresholds.get(level.lower(), 50.0)
    
    def _init_database(self) -> None:
        """Initialize SQLite database with proper schema."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Enable foreign keys
                cursor.execute("PRAGMA foreign_keys = ON")
                
                # Attendance table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS attendance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        name TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        date TEXT NOT NULL,
                        time TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        quality REAL NOT NULL,
                        is_live BOOLEAN NOT NULL,
                        check_type TEXT NOT NULL CHECK(check_type IN ('IN', 'OUT')),
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes for better performance
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_date ON attendance(date)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_name ON attendance(name)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_timestamp ON attendance(timestamp)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_user_id ON attendance(user_id)
                """)
                
                # System info table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS system_info (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Store version
                cursor.execute("""
                    INSERT OR REPLACE INTO system_info (key, value)
                    VALUES ('db_version', ?)
                """, (str(self.DATABASE_VERSION),))
                
                # Auto-commit on context exit
                
            self.logger.info("   ✅ Database initialized")
            
        except Exception as e:
            self.logger.error(f"   ❌ Database initialization failed: {e}")
            raise
    
    def _load_today_records(self) -> None:
        """Load today's attendance records from database."""
        today = datetime.now().strftime('%Y-%m-%d')
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT user_id, name, timestamp, confidence, quality, is_live, check_type
                    FROM attendance
                    WHERE date = ?
                    ORDER BY timestamp ASC
                """, (today,))
                
                rows = cursor.fetchall()
            
            for row in rows:
                timestamp = datetime.fromisoformat(row[2])
                record = AttendanceRecord(
                    user_id=row[0],
                    name=row[1],
                    timestamp=timestamp,
                    confidence=row[3],
                    quality=row[4],
                    is_live=bool(row[5]),
                    check_type=row[6]
                )
                self.attendance_records.append(record)
                
                # Update the appropriate last check time based on type
                if row[6] == "IN":
                    self.last_check_in[row[1]] = timestamp
                else:
                    self.last_check_out[row[1]] = timestamp
            
            if self.attendance_records:
                self.logger.info(f"   📋 Loaded {len(self.attendance_records)} records from today")
            
        except Exception as e:
            self.logger.warning(f"   ⚠️  Error loading records: {e}")
    
    def process_frame(
        self,
        frame: np.ndarray,
        check_type: str = "IN"
    ) -> Tuple[np.ndarray, Optional[AttendanceRecord]]:
        """
        Process a single frame for attendance with comprehensive error handling.
        
        Args:
            frame: Input video frame (BGR format)
            check_type: "IN" or "OUT"
            
        Returns:
            Tuple of (annotated_frame, attendance_record)
            attendance_record is None if no valid check was performed
        """
        try:
            # Validate input
            if frame is None or frame.size == 0:
                raise ValueError("Invalid frame: empty or None")
            
            if check_type not in ["IN", "OUT"]:
                raise ValueError(f"Invalid check_type: {check_type}")
            
            # Periodic cleanup
            if datetime.now() - self._last_cleanup > self._cleanup_interval:
                self._cleanup_old_records()
                self._last_cleanup = datetime.now()
            
            annotated = frame.copy()
            record = None
            
            # Detect faces
            try:
                faces = self.detector.detect_faces(frame, enable_filtering=True)
            except Exception as e:
                self.logger.error(f"Face detection error: {e}")
                cv2.putText(annotated, "Face detection error", (10, 95),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return annotated, None
            
            if len(faces) == 0:
                cv2.putText(annotated, "No face detected", (10, 95),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                return annotated, None
            
            # Process first face only
            x, y, w, h = faces[0]
            
            # Validate face region
            if y < 0 or x < 0 or y+h > frame.shape[0] or x+w > frame.shape[1]:
                cv2.putText(annotated, "Invalid face region", (10, 95),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return annotated, None
            
            face_roi = frame[y:y+h, x:x+w]
            
            # Draw face rectangle
            cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Recognize face
            try:
                result = self.recognizer.predict_with_name(face_roi)
            except Exception as e:
                self.logger.error(f"Recognition error: {e}")
                cv2.putText(annotated, "Recognition error", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                return annotated, None
            
            # Process result
            if result.recognized and result.is_live:
                # Check for duplicate
                if self._is_duplicate_check(result.name, check_type):
                    last_check_dict = self.last_check_in if check_type == "IN" else self.last_check_out
                    time_since = datetime.now() - last_check_dict[result.name]
                    minutes = int(time_since.total_seconds() / 60)
                    wait_time = self.duplicate_threshold.seconds // 60 - minutes
                    
                    cv2.putText(annotated, f"{result.name} - Already {check_type}",
                               (x, y-35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                    cv2.putText(annotated, f"Wait {max(0, wait_time)} min",
                               (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                else:
                    # Valid check-in/out
                    record = self._record_attendance(result, check_type)
                    
                    # Visual feedback
                    color = (0, 255, 0)
                    cv2.putText(annotated, f"✓ {result.name} - {check_type}",
                               (x, y-35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(annotated, f"Attendance recorded: {check_type}",
                               (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            elif not result.is_live:
                # Spoofing detected
                attack_type = result.liveness_result.attack_type.value if result.liveness_result else "Unknown"
                cv2.putText(annotated, f"⚠ SPOOF: {attack_type}", (x, y-35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(annotated, "Anti-spoofing alert!", (10, 95),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            else:
                # Not recognized
                cv2.putText(annotated, f"Unknown (conf: {result.confidence:.1f})", (x, y-35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(annotated, "Person not registered", (10, 95),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show detailed info
            info_y = 125
            cv2.putText(annotated, f"Confidence: {result.confidence:.1f}", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(annotated, f"Quality: {result.quality_score:.2f}", (10, info_y+25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(annotated, f"Live: {'Yes' if result.is_live else 'No'}", (10, info_y+50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return annotated, record
            
        except Exception as e:
            self.logger.error(f"Frame processing error: {e}", exc_info=True)
            # Return original frame with error message
            try:
                cv2.putText(frame, f"Error: {str(e)[:40]}", (10, 95),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            except:
                pass
            return frame, None
    
    def _is_duplicate_check(self, name: str, check_type: str) -> bool:
        """Check if this is a duplicate check within threshold for the same type."""
        # Get the last check time for this specific type
        last_check_dict = self.last_check_in if check_type == "IN" else self.last_check_out
        
        if name not in last_check_dict:
            return False
        
        time_since_last = datetime.now() - last_check_dict[name]
        return time_since_last < self.duplicate_threshold
    
    def _record_attendance(self, result, check_type: str) -> AttendanceRecord:
        """
        Record attendance to database, CSV, and memory with full error handling.
        
        Args:
            result: Recognition result from FaceRecognizer
            check_type: "IN" or "OUT"
            
        Returns:
            AttendanceRecord object
            
        Raises:
            Exception: If recording fails
        """
        now = datetime.now()
        
        try:
            record = AttendanceRecord(
                user_id=result.label,
                name=result.name,
                timestamp=now,
                confidence=result.confidence,
                quality=result.quality_score,
                is_live=result.is_live,
                check_type=check_type
            )
            
            # Add to memory
            self.attendance_records.append(record)
            
            # Update the appropriate last check time based on type
            if check_type == "IN":
                self.last_check_in[result.name] = now
            else:
                self.last_check_out[result.name] = now
            
            # Update daily stats
            stats = self.daily_stats[result.name]
            if check_type == "IN":
                stats['check_ins'] += 1
                stats['last_check_in_time'] = now
            else:
                stats['check_outs'] += 1
                if stats['last_check_in_time']:
                    duration = now - stats['last_check_in_time']
                    stats['total_time'] += duration
            
            # Save to database (thread-safe)
            self._save_to_database(record)
            
            # Backup to CSV
            self._save_to_csv(record)
            
            # Log to report generator
            if self.report_generator:
                try:
                    self.report_generator.log_recognition_attempt(
                        user_name=result.name,
                        recognized=True,
                        confidence=result.confidence,
                        quality_score=result.quality_score,
                        processing_time=0.1,
                        is_spoof=not result.is_live,
                        spoof_type=result.liveness_result.attack_type.value if result.liveness_result else None
                    )
                except Exception as e:
                    self.logger.warning(f"Report logging failed: {e}")
            
            self.logger.info(f"✅ {result.name} - {check_type} at {now.strftime('%H:%M:%S')}")
            
            return record
            
        except Exception as e:
            self.logger.error(f"Recording error: {e}", exc_info=True)
            raise
    
    def _save_to_database(self, record: AttendanceRecord) -> None:
        """Save record to SQLite database with thread safety and transaction handling."""
        with self._db_lock:  # Thread-safe access
            conn = None
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO attendance (user_id, name, timestamp, date, time,
                                           confidence, quality, is_live, check_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.user_id,
                    record.name,
                    record.timestamp.isoformat(),
                    record.timestamp.strftime('%Y-%m-%d'),
                    record.timestamp.strftime('%H:%M:%S'),
                    record.confidence,
                    record.quality,
                    record.is_live,
                    record.check_type
                ))
                
                conn.commit()
                
            except Exception as e:
                if conn:
                    conn.rollback()
                self.logger.error(f"Database save error: {e}")
                raise
                
            finally:
                if conn:
                    conn.close()
    
    def _save_to_csv(self, record: AttendanceRecord) -> None:
        """Save record to CSV backup file."""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            log_file = self.data_dir / "logs" / f"attendance_{today}.csv"
            
            file_exists = log_file.exists()
            
            with open(log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'user_id', 'name', 'timestamp', 'date', 'time',
                    'confidence', 'quality', 'is_live', 'check_type'
                ])
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow(record.to_dict())
                
        except Exception as e:
            self.logger.warning(f"CSV save error: {e}")
    
    def _cleanup_old_records(self) -> None:
        """Clean up old records from memory to prevent memory issues."""
        if len(self.attendance_records) > self.MAX_RECORDS_IN_MEMORY:
            today = datetime.now().date()
            old_count = len(self.attendance_records)
            
            self.attendance_records = [
                r for r in self.attendance_records 
                if r.timestamp.date() == today
            ]
            
            removed = old_count - len(self.attendance_records)
            if removed > 0:
                self.logger.info(f"🧹 Cleaned {removed} old records from memory")
    
    def export_attendance(
        self,
        date: Optional[str] = None,
        format: str = "excel"
    ) -> str:
        """
        Export attendance data with proper error handling.
        
        Args:
            date: Date to export (YYYY-MM-DD) or None for today
            format: Export format (excel/csv/json)
            
        Returns:
            Path to exported file
            
        Raises:
            FileNotFoundError: If no data for the specified date
            ImportError: If required packages not installed
            ValueError: If invalid format
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            # Fetch from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT user_id, name, timestamp, date, time,
                           confidence, quality, is_live, check_type
                    FROM attendance
                    WHERE date = ?
                    ORDER BY timestamp ASC
                """, (date,))
                
                rows = cursor.fetchall()
            
            if not rows:
                raise FileNotFoundError(f"No attendance data for {date}")
            
            # Convert to records
            records = []
            for row in rows:
                records.append({
                    'user_id': row[0],
                    'name': row[1],
                    'timestamp': row[2],
                    'date': row[3],
                    'time': row[4],
                    'confidence': row[5],
                    'quality': row[6],
                    'is_live': row[7],
                    'check_type': row[8]
                })
            
            export_path = self.data_dir / "exports"
            
            if format == "excel":
                try:
                    import pandas as pd  # type: ignore[import-not-found]
                except ImportError:
                    raise ImportError(
                        "pandas and openpyxl required for Excel export.\n"
                        "Install with: pip install pandas openpyxl"
                    )
                
                df = pd.DataFrame(records)
                output_file = export_path / f"attendance_{date}.xlsx"
                df.to_excel(output_file, index=False, engine='openpyxl')
            
            elif format == "csv":
                output_file = export_path / f"attendance_{date}.csv"
                with open(output_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=records[0].keys())
                    writer.writeheader()
                    writer.writerows(records)
            
            elif format == "json":
                output_file = export_path / f"attendance_{date}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(records, f, indent=2, ensure_ascii=False)
            
            else:
                raise ValueError(f"Unknown format: {format}. Use 'excel', 'csv', or 'json'")
            
            self.logger.info(f"📊 Exported {len(records)} records to: {output_file}")
            return str(output_file)
            
        except Exception as e:
            self.logger.error(f"Export error: {e}")
            raise
    
    def get_statistics(self, date: Optional[str] = None) -> Dict[str, Any]:
        """Get attendance statistics from database."""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total records
                cursor.execute("""
                    SELECT COUNT(*) FROM attendance WHERE date = ?
                """, (date,))
                total_records = cursor.fetchone()[0]
                
                if total_records == 0:
                    return {
                        'date': date,
                        'total_records': 0,
                        'message': 'No attendance data for this date'
                    }
                
                # Unique people
                cursor.execute("""
                    SELECT DISTINCT name FROM attendance WHERE date = ?
                """, (date,))
                unique_people = [row[0] for row in cursor.fetchall()]
                
                # Check-ins and check-outs
                cursor.execute("""
                    SELECT check_type, COUNT(*) FROM attendance
                    WHERE date = ? GROUP BY check_type
                """, (date,))
                check_counts = dict(cursor.fetchall())
                
                # Average metrics
                cursor.execute("""
                    SELECT AVG(confidence), AVG(quality)
                    FROM attendance WHERE date = ?
                """, (date,))
                avg_conf, avg_qual = cursor.fetchone()
                
                # Time range
                cursor.execute("""
                    SELECT MIN(time), MAX(time)
                    FROM attendance WHERE date = ?
                """, (date,))
                first_time, last_time = cursor.fetchone()
            
            return {
                'date': date,
                'total_records': total_records,
                'unique_people': len(unique_people),
                'people': unique_people,
                'check_ins': check_counts.get('IN', 0),
                'check_outs': check_counts.get('OUT', 0),
                'avg_confidence': round(avg_conf, 2) if avg_conf else 0,
                'avg_quality': round(avg_qual, 3) if avg_qual else 0,
                'first_check_in': first_time,
                'last_check_out': last_time
            }
            
        except Exception as e:
            self.logger.error(f"Statistics error: {e}")
            return {'date': date, 'error': str(e)}
    
    def generate_reports(self) -> str:
        """Generate comprehensive user reports."""
        if not self.report_generator:
            return "Reports disabled"
        
        try:
            report = self.report_generator.finalize_session()
            user_count = len(report['per_user_details'])
            self.logger.info(f"📊 Reports generated for {user_count} users")
            return f"Reports generated: {user_count} users"
        except Exception as e:
            self.logger.error(f"Report generation error: {e}")
            return f"Error: {e}"
    
    def run_live_mode(self, check_type: str = "IN") -> None:
        """Run live attendance capture with comprehensive error handling."""
        self.logger.info("="*70)
        self.logger.info(f"🎥 LIVE ATTENDANCE MODE - {check_type}")
        self.logger.info("="*70)
        self.logger.info("Controls:")
        self.logger.info("  'i' - Switch to CHECK-IN mode")
        self.logger.info("  'o' - Switch to CHECK-OUT mode")
        self.logger.info("  's' - Show statistics")
        self.logger.info("  'e' - Export today's data")
        self.logger.info("  'q' - Quit and generate reports")
        self.logger.info("="*70)
        
        cap = None
        
        try:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                raise RuntimeError("Failed to open camera. Check connection.")
            
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            current_mode = check_type
            frame_count = 0
            
            self.logger.info("Camera opened successfully. Starting capture...")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning("Failed to read frame")
                    time.sleep(0.1)
                    continue
                
                frame_count += 1
                
                # Process frame
                annotated, record = self.process_frame(frame, current_mode)
                
                # Draw UI elements
                h, w = annotated.shape[:2]
                
                # Mode indicator
                mode_color = (0, 255, 0) if current_mode == "IN" else (0, 165, 255)
                cv2.rectangle(annotated, (0, h-60), (250, h), mode_color, -1)
                cv2.putText(annotated, f"Mode: CHECK-{current_mode}", 
                           (10, h-35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(annotated, f"Records: {len(self.attendance_records)}", 
                           (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                # Controls help
                cv2.putText(annotated, "i:IN | o:OUT | s:Stats | e:Export | q:Quit", 
                           (10, h-70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show window
                cv2.imshow('Attendance System v2.0', annotated)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    self.logger.info("Quit requested")
                    break
                    
                elif key == ord('i'):
                    current_mode = "IN"
                    self.logger.info("✓ Switched to CHECK-IN mode")
                    
                elif key == ord('o'):
                    current_mode = "OUT"
                    self.logger.info("✓ Switched to CHECK-OUT mode")
                    
                elif key == ord('s'):
                    self._print_statistics()
                    
                elif key == ord('e'):
                    try:
                        output = self.export_attendance(format='excel')
                        self.logger.info(f"✅ Exported to: {output}")
                    except Exception as e:
                        self.logger.error(f"❌ Export failed: {e}")
        
        except KeyboardInterrupt:
            self.logger.warning("Interrupted by user")
            
        except Exception as e:
            self.logger.error(f"Error in live mode: {e}", exc_info=True)
            
        finally:
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()
            
            # Always generate reports
            self.logger.info("📊 Generating final reports...")
            try:
                self.generate_reports()
            except Exception as e:
                self.logger.warning(f"Report generation failed: {e}")
            
            self.logger.info("✅ Session ended")
    
    def _print_statistics(self) -> None:
        """Print current session statistics to console."""
        stats = self.get_statistics()
        
        print(f"\n{'='*70}")
        print("📊 TODAY'S STATISTICS")
        print(f"{'='*70}")
        
        if stats.get('total_records', 0) == 0:
            print("  No records yet today")
        else:
            print(f"  Date:              {stats['date']}")
            print(f"  Total Records:     {stats['total_records']}")
            print(f"  Unique People:     {stats['unique_people']}")
            print(f"  Check-INs:         {stats['check_ins']}")
            print(f"  Check-OUTs:        {stats['check_outs']}")
            print(f"  Avg Confidence:    {stats['avg_confidence']:.2f}")
            print(f"  Avg Quality:       {stats['avg_quality']:.3f}")
            print(f"  First Check-IN:    {stats.get('first_check_in', 'N/A')}")
            print(f"  Last Check-OUT:    {stats.get('last_check_out', 'N/A')}")
            
            if stats.get('people'):
                people_list = ', '.join(stats['people'][:10])
                if len(stats['people']) > 10:
                    people_list += f" ... (+{len(stats['people']) - 10} more)"
                print(f"  People:            {people_list}")
        
        print(f"{'='*70}\n")
    
    def backup_database(self, backup_name: Optional[str] = None) -> str:
        """Create a backup of the database."""
        if backup_name is None:
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        
        backup_path = self.data_dir / "backups" / backup_name
        
        try:
            shutil.copy2(self.db_path, backup_path)
            self.logger.info(f"💾 Database backed up to: {backup_path}")
            return str(backup_path)
        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
            raise
    
    def get_user_report(self, name: str, days: int = 7) -> Dict[str, Any]:
        """Get detailed report for a specific user."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get records for last N days
                start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
                
                cursor.execute("""
                    SELECT date, time, check_type, confidence, quality
                    FROM attendance
                    WHERE name = ? AND date >= ?
                    ORDER BY timestamp DESC
                """, (name, start_date))
                
                records = cursor.fetchall()
            
            if not records:
                return {'name': name, 'records': 0, 'message': f'No records in last {days} days'}
            
            # Calculate statistics
            check_ins = sum(1 for r in records if r[2] == 'IN')
            check_outs = sum(1 for r in records if r[2] == 'OUT')
            avg_confidence = np.mean([r[3] for r in records])
            avg_quality = np.mean([r[4] for r in records])
            
            # Group by date
            dates = {}
            for record in records:
                date = record[0]
                if date not in dates:
                    dates[date] = {'IN': [], 'OUT': []}
                dates[date][record[2]].append(record[1])
            
            return {
                'name': name,
                'period_days': days,
                'total_records': len(records),
                'check_ins': check_ins,
                'check_outs': check_outs,
                'avg_confidence': round(avg_confidence, 2),
                'avg_quality': round(avg_quality, 3),
                'daily_records': dates,
                'recent_records': [
                    {'date': r[0], 'time': r[1], 'type': r[2]} 
                    for r in records[:10]
                ]
            }
            
        except Exception as e:
            self.logger.error(f"User report error: {e}")
            return {'name': name, 'error': str(e)}


def main() -> None:
    """Main entry point with improved CLI and error handling."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Perfect Attendance Management System v2.0 - Production Ready',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run live attendance mode (default)
  python attendance_system.py
  
  # Run with specific security level
  python attendance_system.py --mode live --security strict
  
  # View today's statistics
  python attendance_system.py --mode stats
  
  # View specific date statistics
  python attendance_system.py --mode stats --date 2025-01-15
  
  # Export today's data to Excel
  python attendance_system.py --mode export --format excel
  
  # Export specific date to CSV
  python attendance_system.py --mode export --date 2025-01-15 --format csv
  
  # Get user report
  python attendance_system.py --mode user --name "John Doe" --days 7
  
  # Backup database
  python attendance_system.py --mode backup

For more information, visit: https://github.com/OmarBadrawyyy/Anti-spoof-facial-recognition
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['live', 'stats', 'export', 'user', 'backup'],
        default='live',
        help='Operation mode (default: live)'
    )
    parser.add_argument(
        '--model',
        default='models/combined_model.yml',
        help='Path to recognition model (default: models/combined_model.yml)'
    )
    parser.add_argument(
        '--security',
        choices=['lenient', 'balanced', 'strict'],
        default='balanced',
        help='Security level (default: balanced)'
    )
    parser.add_argument(
        '--date',
        help='Date for stats/export (YYYY-MM-DD format)'
    )
    parser.add_argument(
        '--format',
        choices=['excel', 'csv', 'json'],
        default='excel',
        help='Export format (default: excel)'
    )
    parser.add_argument(
        '--name',
        help='User name for user report mode'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Number of days for user report (default: 7)'
    )
    parser.add_argument(
        '--data-dir',
        default='attendance_data',
        help='Data directory (default: attendance_data)'
    )
    parser.add_argument(
        '--no-antispoofing',
        action='store_true',
        help='Disable anti-spoofing detection'
    )
    parser.add_argument(
        '--no-reports',
        action='store_true',
        help='Disable per-user reporting'
    )
    parser.add_argument(
        '--version',
        action='version',
        version='Attendance System v2.0.0'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize system
        print("Initializing system...")
        system = AttendanceManager(
            model_path=args.model,
            data_dir=args.data_dir,
            security_level=args.security,
            enable_antispoofing=not args.no_antispoofing,
            enable_reports=not args.no_reports
        )
        
        # Execute mode
        if args.mode == 'live':
            system.run_live_mode()
        
        elif args.mode == 'stats':
            stats = system.get_statistics(args.date)
            
            print(f"\n{'='*70}")
            print("📊 ATTENDANCE STATISTICS")
            print(f"{'='*70}")
            
            for key, value in stats.items():
                if isinstance(value, list):
                    value_str = ', '.join(str(v) for v in value[:10])
                    if len(value) > 10:
                        value_str += f" ... (+{len(value) - 10} more)"
                    print(f"  {key:20}: {value_str}")
                else:
                    print(f"  {key:20}: {value}")
            
            print(f"{'='*70}\n")
        
        elif args.mode == 'export':
            output = system.export_attendance(args.date, args.format)
            print(f"\n✅ Export successful!")
            print(f"   File: {output}")
        
        elif args.mode == 'user':
            if not args.name:
                print("❌ Error: --name required for user mode")
                print("Example: python attendance_system.py --mode user --name \"John Doe\"")
                sys.exit(1)
            
            report = system.get_user_report(args.name, args.days)
            
            print(f"\n{'='*70}")
            print(f"👤 USER REPORT: {args.name}")
            print(f"{'='*70}")
            
            for key, value in report.items():
                if key not in ['daily_records', 'recent_records']:
                    print(f"  {key:20}: {value}")
            
            if 'recent_records' in report:
                print(f"\n  Recent Activity:")
                for record in report['recent_records'][:5]:
                    print(f"    {record['date']} {record['time']} - {record['type']}")
            
            print(f"{'='*70}\n")
        
        elif args.mode == 'backup':
            backup_path = system.backup_database()
            print(f"\n✅ Backup created successfully!")
            print(f"   Path: {backup_path}")
        
        sys.exit(0)
    
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        sys.exit(1)
        
    except ImportError as e:
        print(f"\n❌ Missing dependency: {e}")
        print("\nInstall required packages:")
        print("  pip install opencv-python numpy pandas openpyxl")
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
