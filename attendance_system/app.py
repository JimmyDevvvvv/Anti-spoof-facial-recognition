"""
Perfect Attendance System - Enhanced Flask REST API Backend
==================================================

Production-ready Flask backend with comprehensive API endpoints
"""

import base64
import csv
import io
import json
import logging
import os
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# Add parent directory to path for imports
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent if current_file.parent.name == "attendance_system" else current_file.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import attendance system components
try:
    from src.face.detector import FaceDetector
    from src.face.recognizer import FaceRecognizer
    from src.face.final_anti_spoof import UltimateAntiSpoof
    from src.face.per_user_report_generator import PerUserReportGenerator
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Please ensure all required modules are in src/face/")
    sys.exit(1)

# ============================================================================
# ATTENDANCE RECORD CLASS
# ============================================================================

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
        """Initialize attendance record with validation."""
        if check_type not in ["IN", "OUT"]:
            raise ValueError(f"check_type must be 'IN' or 'OUT', got: {check_type}")
        
        if not name or not isinstance(name, str):
            raise ValueError("name must be a non-empty string")
        
        if not isinstance(user_id, int) or user_id < 0:
            raise ValueError("user_id must be a non-negative integer")
        
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


# ============================================================================
# ATTENDANCE MANAGER CLASS
# ============================================================================

class AttendanceManager:
    """Core attendance management system."""
    
    MAX_RECORDS_IN_MEMORY = 1000
    
    def __init__(
        self,
        model_path: str = "models/combined_model.yml",
        data_dir: str = "attendance_data",
        security_level: str = "balanced",
        duplicate_threshold_minutes: int = 5,
        enable_antispoofing: bool = True,
        enable_reports: bool = True
    ):
        """Initialize attendance management system."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        for subdir in ['logs', 'reports', 'exports', 'backups']:
            (self.data_dir / subdir).mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.security_level = security_level
        self.duplicate_threshold = timedelta(minutes=duplicate_threshold_minutes)
        self._db_lock = Lock()
        
        self.logger.info("🚀 Initializing Attendance Management System")
        
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
        else:
            self.logger.warning(f"   ⚠️  No model found at: {model_path}")
        
        # Anti-spoofing detector
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
        
        # Report generator
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
        
        # Initialize database
        self.db_path = self.data_dir / "attendance.db"
        self._init_database()
        
        # Session tracking
        self.attendance_records: List[AttendanceRecord] = []
        self.last_check_in: Dict[str, datetime] = {}
        self.last_check_out: Dict[str, datetime] = {}
        
        # Load today's records
        self._load_today_records()
        
        self.logger.info("✅ System ready!")
    
    def _get_threshold_for_level(self, level: str) -> float:
        """Get recognition threshold for security level."""
        thresholds = {
            'lenient': 60.0,
            'balanced': 50.0,
            'strict': 40.0
        }
        return thresholds.get(level.lower(), 50.0)
    
    def _init_database(self) -> None:
        """Initialize SQLite database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("PRAGMA foreign_keys = ON")
                
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
                
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_date ON attendance(date)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_name ON attendance(name)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON attendance(timestamp)")
            
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
        """Process a single frame for attendance."""
        try:
            if frame is None or frame.size == 0:
                raise ValueError("Invalid frame: empty or None")
            
            if check_type not in ["IN", "OUT"]:
                raise ValueError(f"Invalid check_type: {check_type}")
            
            annotated = frame.copy()
            record = None
            
            # Detect faces
            faces = self.detector.detect_faces(frame, enable_filtering=True)
            
            if len(faces) == 0:
                cv2.putText(annotated, "No face detected", (10, 95),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                return annotated, None
            
            # Process first face only
            x, y, w, h = faces[0]
            
            if y < 0 or x < 0 or y+h > frame.shape[0] or x+w > frame.shape[1]:
                cv2.putText(annotated, "Invalid face region", (10, 95),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return annotated, None
            
            face_roi = frame[y:y+h, x:x+w]
            
            # Draw face rectangle
            cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Recognize face
            result = self.recognizer.predict_with_name(face_roi)
            
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
            try:
                cv2.putText(frame, f"Error: {str(e)[:40]}", (10, 95),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            except:
                pass
            return frame, None
    
    def _is_duplicate_check(self, name: str, check_type: str) -> bool:
        """Check if this is a duplicate check within threshold."""
        last_check_dict = self.last_check_in if check_type == "IN" else self.last_check_out
        
        if name not in last_check_dict:
            return False
        
        time_since_last = datetime.now() - last_check_dict[name]
        return time_since_last < self.duplicate_threshold
    
    def _record_attendance(self, result, check_type: str) -> AttendanceRecord:
        """Record attendance to database and memory."""
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
            
            # Update last check time
            if check_type == "IN":
                self.last_check_in[result.name] = now
            else:
                self.last_check_out[result.name] = now
            
            # Save to database
            self._save_to_database(record)
            
            # Backup to CSV
            self._save_to_csv(record)
            
            self.logger.info(f"✅ {result.name} - {check_type} at {now.strftime('%H:%M:%S')}")
            
            return record
            
        except Exception as e:
            self.logger.error(f"Recording error: {e}", exc_info=True)
            raise
    
    def _save_to_database(self, record: AttendanceRecord) -> None:
        """Save record to SQLite database."""
        with self._db_lock:
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
    
    def get_statistics(self, date: Optional[str] = None) -> Dict[str, Any]:
        """Get attendance statistics from database."""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) FROM attendance WHERE date = ?", (date,))
                total_records = cursor.fetchone()[0]
                
                if total_records == 0:
                    return {'date': date, 'total_records': 0, 'message': 'No attendance data for this date'}
                
                cursor.execute("SELECT DISTINCT name FROM attendance WHERE date = ?", (date,))
                unique_people = [row[0] for row in cursor.fetchall()]
                
                cursor.execute("""
                    SELECT check_type, COUNT(*) FROM attendance
                    WHERE date = ? GROUP BY check_type
                """, (date,))
                check_counts = dict(cursor.fetchall())
                
                cursor.execute("SELECT AVG(confidence), AVG(quality) FROM attendance WHERE date = ?", (date,))
                avg_conf, avg_qual = cursor.fetchone()
                
                cursor.execute("SELECT MIN(time), MAX(time) FROM attendance WHERE date = ?", (date,))
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
    
    def get_date_range_records(self, start_date: str, end_date: str) -> List[Dict]:
        """Get records for date range."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT user_id, name, timestamp, date, time, confidence, quality, is_live, check_type
                    FROM attendance
                    WHERE date BETWEEN ? AND ?
                    ORDER BY timestamp DESC
                """, (start_date, end_date))
                
                rows = cursor.fetchall()
                return [
                    {
                        'user_id': r[0], 'name': r[1], 'timestamp': r[2],
                        'date': r[3], 'time': r[4], 'confidence': r[5],
                        'quality': r[6], 'is_live': r[7], 'check_type': r[8]
                    }
                    for r in rows
                ]
        except Exception as e:
            self.logger.error(f"Date range query error: {e}")
            return []
    
    def export_attendance(self, date: Optional[str] = None, format: str = "excel") -> str:
        """Export attendance data."""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT user_id, name, timestamp, date, time,
                           confidence, quality, is_live, check_type
                    FROM attendance WHERE date = ? ORDER BY timestamp ASC
                """, (date,))
                
                rows = cursor.fetchall()
            
            if not rows:
                raise FileNotFoundError(f"No attendance data for {date}")
            
            records = []
            for row in rows:
                records.append({
                    'user_id': row[0], 'name': row[1], 'timestamp': row[2],
                    'date': row[3], 'time': row[4], 'confidence': row[5],
                    'quality': row[6], 'is_live': row[7], 'check_type': row[8]
                })
            
            export_path = self.data_dir / "exports"
            
            if format == "excel":
                import pandas as pd
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
                raise ValueError(f"Unknown format: {format}")
            
            self.logger.info(f"📊 Exported {len(records)} records to: {output_file}")
            return str(output_file)
        except Exception as e:
            self.logger.error(f"Export error: {e}")
            raise
    
    def backup_database(self, backup_name: Optional[str] = None) -> str:
        """Create a backup of the database."""
        import shutil
        
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
                
                start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
                
                cursor.execute("""
                    SELECT date, time, check_type, confidence, quality
                    FROM attendance WHERE name = ? AND date >= ?
                    ORDER BY timestamp DESC
                """, (name, start_date))
                
                records = cursor.fetchall()
            
            if not records:
                return {'name': name, 'records': 0, 'message': f'No records in last {days} days'}
            
            check_ins = sum(1 for r in records if r[2] == 'IN')
            check_outs = sum(1 for r in records if r[2] == 'OUT')
            avg_confidence = np.mean([r[3] for r in records])
            avg_quality = np.mean([r[4] for r in records])
            
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


# ============================================================================
# FLASK APP INITIALIZATION
# ============================================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "http://localhost:3001"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

socketio = SocketIO(
    app,
    cors_allowed_origins=["http://localhost:3000", "http://localhost:3001"],
    async_mode='threading'
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# GLOBAL STATE
# ============================================================================

attendance_manager = None
camera = None
current_mode: str = "IN"
camera_lock = Lock()
system_lock = Lock()
camera_active = False
last_notification_time = 0
notification_cooldown = 1.0

# ============================================================================
# INITIALIZATION
# ============================================================================

def init_system(model_path: str = "models/combined_model.yml", security_level: str = "balanced"):
    """Initialize the attendance system."""
    global attendance_manager
    
    with system_lock:
        if attendance_manager is None:
            logger.info("🚀 Initializing Attendance System...")
            try:
                attendance_manager = AttendanceManager(
                    model_path=model_path,
                    data_dir="attendance_data",
                    security_level=security_level,
                    enable_antispoofing=True,
                    enable_reports=True
                )
                logger.info("✅ System initialized successfully!")
            except Exception as e:
                logger.error(f"❌ System initialization failed: {e}")
                raise


def init_camera():
    """Initialize camera with proper settings."""
    global camera, camera_active
    
    with camera_lock:
        if camera is None:
            try:
                camera = cv2.VideoCapture(0)
                if not camera.isOpened():
                    raise RuntimeError("Failed to open camera")
                
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                camera.set(cv2.CAP_PROP_FPS, 30)
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                camera_active = True
                logger.info("📷 Camera initialized successfully")
            except Exception as e:
                logger.error(f"❌ Camera initialization failed: {e}")
                camera = None
                camera_active = False
                raise


def release_camera():
    """Release camera resources."""
    global camera, camera_active
    
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
            camera_active = False
            logger.info("📷 Camera released")


# ============================================================================
# WEBSOCKET HELPERS
# ============================================================================

def emit_notification(notification_type: str, title: str, message: str, data: Optional[Dict] = None):
    """Emit real-time notification to all connected clients."""
    global last_notification_time
    
    current_time = time.time()
    if current_time - last_notification_time < notification_cooldown:
        return
    
    last_notification_time = current_time
    
    notification = {
        'type': notification_type,
        'title': title,
        'message': message,
        'timestamp': datetime.now().isoformat(),
        'data': data or {}
    }
    
    socketio.emit('notification', notification)
    logger.info(f"📢 Notification: {title}")


def emit_stats_update():
    """Emit statistics update to all clients."""
    if attendance_manager:
        stats = get_quick_stats()
        socketio.emit('stats_update', stats)


def emit_record_update(record):
    """Emit new record to all clients."""
    socketio.emit('new_record', record.to_dict())
    emit_stats_update()


# ============================================================================
# VIDEO STREAMING
# ============================================================================

def generate_frames():
    """Generate video frames with face recognition."""
    global current_mode, camera_active
    
    if not attendance_manager:
        logger.error("Attendance system not initialized")
        return
    
    if not camera_active:
        init_camera()
    
    frame_count = 0
    last_stats_update = time.time()
    
    while camera_active:
        with camera_lock:
            if camera is None:
                break
            
            success, frame = camera.read()
        
        if not success:
            logger.warning("Failed to read frame")
            time.sleep(0.1)
            continue
        
        frame_count += 1
        
        try:
            annotated, record = attendance_manager.process_frame(frame, current_mode)
            
            if record is not None:
                emit_record_update(record)
                
                if record.is_live:
                    emit_notification(
                        'success',
                        f'✓ {record.name} Checked {record.check_type}',
                        f'Time: {record.timestamp.strftime("%H:%M:%S")} | Confidence: {record.confidence:.1f}',
                        record.to_dict()
                    )
                else:
                    emit_notification(
                        'error',
                        '⚠️ Spoofing Detected!',
                        f'Attempted check by {record.name} was BLOCKED',
                        record.to_dict()
                    )
            
            if time.time() - last_stats_update > 5:
                emit_stats_update()
                last_stats_update = time.time()
            
            h, w = annotated.shape[:2]
            
            mode_color = (0, 255, 0) if current_mode == "IN" else (255, 140, 0)
            cv2.rectangle(annotated, (10, 10), (250, 80), mode_color, -1)
            cv2.putText(annotated, f"Mode: CHECK-{current_mode}",
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(annotated, f"Records: {len(attendance_manager.attendance_records)}",
                       (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            cv2.putText(annotated, f"Frame: {frame_count}",
                       (w - 150, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            ret, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            time.sleep(0.1)


def get_quick_stats() -> Dict[str, Any]:
    """Get quick statistics without database query."""
    if not attendance_manager:
        return {}
    
    records = attendance_manager.attendance_records
    unique_people = set(r.name for r in records)
    check_ins = sum(1 for r in records if r.check_type == 'IN')
    check_outs = sum(1 for r in records if r.check_type == 'OUT')
    
    return {
        'total_records': len(records),
        'unique_people': len(unique_people),
        'check_ins': check_ins,
        'check_outs': check_outs,
        'current_mode': current_mode
    }


# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'system_initialized': attendance_manager is not None,
        'camera_active': camera_active
    })


@app.route('/api/system/info', methods=['GET'])
def system_info():
    """Get system information."""
    if not attendance_manager:
        return jsonify({'error': 'System not initialized'}), 503
    
    try:
        model_info = attendance_manager.recognizer.get_model_info()
        
        return jsonify({
            'security_level': attendance_manager.security_level,
            'antispoofing_enabled': attendance_manager.antispoofing is not None,
            'reports_enabled': attendance_manager.report_generator is not None,
            'model_info': model_info,
            'current_mode': current_mode,
            'camera_active': camera_active,
            'version': '1.0.0'
        })
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/video/feed')
def video_feed():
    """Video streaming endpoint."""
    if not attendance_manager:
        return jsonify({'error': 'System not initialized'}), 503
    
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/mode', methods=['GET'])
def get_mode():
    """Get current mode."""
    return jsonify({'mode': current_mode})


@app.route('/api/mode', methods=['POST'])
def set_mode():
    """Set attendance mode."""
    global current_mode
    
    data = request.json or {}
    new_mode = data.get('mode', '').upper()
    
    if new_mode not in ['IN', 'OUT']:
        return jsonify({'error': 'Invalid mode'}), 400
    
    current_mode = new_mode
    socketio.emit('mode_changed', {'mode': current_mode})
    
    logger.info(f"Mode changed to: {current_mode}")
    return jsonify({'success': True, 'mode': current_mode})


@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get attendance statistics."""
    if not attendance_manager:
        return jsonify({'error': 'System not initialized'}), 503
    
    date = request.args.get('date')
    
    try:
        stats = attendance_manager.get_statistics(date)
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Statistics error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/statistics/quick', methods=['GET'])
def quick_statistics():
    """Get quick statistics."""
    if not attendance_manager:
        return jsonify({'error': 'System not initialized'}), 503
    
    return jsonify(get_quick_stats())


@app.route('/api/records/today', methods=['GET'])
def get_today_records():
    """Get today's records."""
    if not attendance_manager:
        return jsonify({'error': 'System not initialized'}), 503
    
    records = [r.to_dict() for r in attendance_manager.attendance_records]
    return jsonify({'records': records, 'count': len(records)})


@app.route('/api/records/latest', methods=['GET'])
def get_latest_records():
    """Get latest N records."""
    if not attendance_manager:
        return jsonify({'error': 'System not initialized'}), 503
    
    n = int(request.args.get('n', 10))
    records = [r.to_dict() for r in attendance_manager.attendance_records[-n:]]
    records.reverse()
    
    return jsonify({'records': records, 'count': len(records)})


@app.route('/api/records/range', methods=['GET'])
def get_date_range_records():
    """Get records for date range."""
    if not attendance_manager:
        return jsonify({'error': 'System not initialized'}), 503
    
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    if not start_date or not end_date:
        return jsonify({'error': 'start_date and end_date required'}), 400
    
    try:
        records = attendance_manager.get_date_range_records(start_date, end_date)
        return jsonify({'records': records, 'count': len(records)})
    except Exception as e:
        logger.error(f"Date range query error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/users', methods=['GET'])
def get_users():
    """Get all registered users."""
    if not attendance_manager:
        return jsonify({'error': 'System not initialized'}), 503
    
    try:
        people = attendance_manager.recognizer.get_known_people()
        return jsonify({'users': people, 'count': len(people)})
    except Exception as e:
        logger.error(f"Users query error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/users/<name>/report', methods=['GET'])
def get_user_report(name: str):
    """Get user report."""
    if not attendance_manager:
        return jsonify({'error': 'System not initialized'}), 503
    
    days = int(request.args.get('days', 7))
    
    try:
        report = attendance_manager.get_user_report(name, days)
        return jsonify(report)
    except Exception as e:
        logger.error(f"User report error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/export', methods=['GET'])
def export_data():
    """Export attendance data."""
    if not attendance_manager:
        return jsonify({'error': 'System not initialized'}), 503
    
    date = request.args.get('date')
    format_type = request.args.get('format', 'excel')
    
    if format_type not in ['excel', 'csv', 'json']:
        return jsonify({'error': 'Invalid format'}), 400
    
    try:
        file_path = attendance_manager.export_attendance(date, format_type)
        return send_file(file_path, as_attachment=True)
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Export error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/backup', methods=['POST'])
def create_backup():
    """Create database backup."""
    if not attendance_manager:
        return jsonify({'error': 'System not initialized'}), 503
    
    try:
        backup_path = attendance_manager.backup_database()
        return jsonify({
            'success': True,
            'path': backup_path,
            'message': 'Backup created successfully'
        })
    except Exception as e:
        logger.error(f"Backup error: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# WEBSOCKET EVENTS
# ============================================================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info(f"🔌 Client connected")
    
    if attendance_manager:
        emit('stats_update', get_quick_stats())
        emit('mode_changed', {'mode': current_mode})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info(f"🔌 Client disconnected")


@socketio.on('request_stats')
def handle_stats_request():
    """Handle stats request."""
    if attendance_manager:
        emit('stats_update', get_quick_stats())


@socketio.on('request_records')
def handle_records_request(data):
    """Handle records request."""
    if attendance_manager:
        n = data.get('count', 10)
        records = [r.to_dict() for r in attendance_manager.attendance_records[-n:]]
        records.reverse()
        emit('records_update', {'records': records})


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Attendance System Flask Backend')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--model', default='models/combined_model.yml', help='Model path')
    parser.add_argument('--security', default='balanced', choices=['lenient', 'balanced', 'strict'])
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🚀 ATTENDANCE SYSTEM - FLASK BACKEND API")
    print("="*70)
    print(f"   URL: http://{args.host}:{args.port}")
    print(f"   Security: {args.security.upper()}")
    print("="*70 + "\n")
    
    try:
        init_system(model_path=args.model, security_level=args.security)
        socketio.run(app, host=args.host, port=args.port, debug=args.debug, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\n\n⚠️ Server stopped")
    finally:
        release_camera()
        print("✅ Cleanup complete")


if __name__ == '__main__':
    main()