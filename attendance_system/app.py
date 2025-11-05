"""
Perfect Attendance System - Enhanced Flask REST API Backend with Authentication
===============================================================================

Production-ready Flask backend with JWT auth, role-based access control
Uses JSON file storage instead of SQLite database
"""

import base64
import csv
import io
import json
import logging
import os
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Dict, List, Optional, Tuple
from functools import wraps

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import jwt
from werkzeug.security import generate_password_hash, check_password_hash

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
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Please ensure all required modules are in src/face/")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
JWT_EXPIRATION_HOURS = 24

# ============================================================================
# JSON STORAGE MANAGER
# ============================================================================

class JSONStorage:
    """Thread-safe JSON file storage manager."""
    
    def __init__(self, base_dir: str = "attendance_data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Storage files
        self.users_file = self.base_dir / "users.json"
        self.attendance_file = self.base_dir / "attendance.json"
        self.settings_file = self.base_dir / "settings.json"
        
        # Locks for thread safety
        self.users_lock = Lock()
        self.attendance_lock = Lock()
        self.settings_lock = Lock()
        
        # Initialize files if they don't exist
        self._init_files()
        
        self.logger = logging.getLogger(__name__)
    
    def _init_files(self):
        """Initialize JSON files with default data."""
        # Users file
        if not self.users_file.exists():
            default_users = {
                "users": [
                    {
                        "id": 1,
                        "username": "admin",
                        "password": generate_password_hash("admin123"),
                        "role": "admin",
                        "name": "System Administrator",
                        "email": "admin@example.com",
                        "created_at": datetime.now().isoformat(),
                        "active": True
                    },
                    {
                        "id": 2,
                        "username": "user",
                        "password": generate_password_hash("user123"),
                        "role": "user",
                        "name": "Regular User",
                        "email": "user@example.com",
                        "created_at": datetime.now().isoformat(),
                        "active": True
                    }
                ]
            }
            self._write_json(self.users_file, default_users, self.users_lock)
        
        # Attendance file
        if not self.attendance_file.exists():
            default_attendance = {
                "records": []
            }
            self._write_json(self.attendance_file, default_attendance, self.attendance_lock)
        
        # Settings file
        if not self.settings_file.exists():
            default_settings = {
                "system": {
                    "security_level": "balanced",
                    "duplicate_threshold_minutes": 5,
                    "enable_antispoofing": True
                },
                "metadata": {
                    "initialized_at": datetime.now().isoformat(),
                    "version": "1.0.0"
                }
            }
            self._write_json(self.settings_file, default_settings, self.settings_lock)
    
    def _read_json(self, file_path: Path, lock: Lock) -> Dict:
        """Thread-safe JSON read."""
        with lock:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error reading {file_path}: {e}")
                return {}
    
    def _write_json(self, file_path: Path, data: Dict, lock: Lock) -> bool:
        """Thread-safe JSON write with backup."""
        with lock:
            try:
                # Create backup
                if file_path.exists():
                    backup_path = file_path.with_suffix('.json.bak')
                    shutil.copy2(file_path, backup_path)
                
                # Write new data
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                return True
            except Exception as e:
                self.logger.error(f"Error writing {file_path}: {e}")
                return False
    
    # User operations
    def get_users(self) -> List[Dict]:
        """Get all users."""
        data = self._read_json(self.users_file, self.users_lock)
        return data.get("users", [])
    
    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """Get user by username."""
        users = self.get_users()
        for user in users:
            if user["username"] == username:
                return user
        return None
    
    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """Get user by ID."""
        users = self.get_users()
        for user in users:
            if user["id"] == user_id:
                return user
        return None
    
    def create_user(self, user_data: Dict) -> Dict:
        """Create new user."""
        data = self._read_json(self.users_file, self.users_lock)
        users = data.get("users", [])
        
        # Generate new ID
        new_id = max([u["id"] for u in users], default=0) + 1
        
        # Create user
        new_user = {
            "id": new_id,
            "username": user_data["username"],
            "password": generate_password_hash(user_data["password"]),
            "role": user_data.get("role", "user"),
            "name": user_data.get("name", user_data["username"]),
            "email": user_data.get("email", ""),
            "created_at": datetime.now().isoformat(),
            "active": True
        }
        
        users.append(new_user)
        data["users"] = users
        
        if self._write_json(self.users_file, data, self.users_lock):
            return new_user
        return {}
    
    def update_user(self, user_id: int, updates: Dict) -> bool:
        """Update user."""
        data = self._read_json(self.users_file, self.users_lock)
        users = data.get("users", [])
        
        for i, user in enumerate(users):
            if user["id"] == user_id:
                # Update allowed fields
                if "name" in updates:
                    users[i]["name"] = updates["name"]
                if "email" in updates:
                    users[i]["email"] = updates["email"]
                if "password" in updates:
                    users[i]["password"] = generate_password_hash(updates["password"])
                if "active" in updates:
                    users[i]["active"] = updates["active"]
                
                users[i]["updated_at"] = datetime.now().isoformat()
                data["users"] = users
                return self._write_json(self.users_file, data, self.users_lock)
        
        return False
    
    def delete_user(self, user_id: int) -> bool:
        """Delete user (soft delete - set active=False)."""
        return self.update_user(user_id, {"active": False})
    
    # Attendance operations
    def get_attendance_records(self, filters: Optional[Dict] = None) -> List[Dict]:
        """Get attendance records with optional filters."""
        data = self._read_json(self.attendance_file, self.attendance_lock)
        records = data.get("records", [])
        
        if not filters:
            return records
        
        # Apply filters
        filtered = records
        
        if "date" in filters:
            filtered = [r for r in filtered if r["date"] == filters["date"]]
        
        if "user_id" in filters:
            filtered = [r for r in filtered if r["user_id"] == filters["user_id"]]
        
        if "name" in filters:
            filtered = [r for r in filtered if r["name"] == filters["name"]]
        
        if "check_type" in filters:
            filtered = [r for r in filtered if r["check_type"] == filters["check_type"]]
        
        if "start_date" in filters and "end_date" in filters:
            filtered = [r for r in filtered 
                       if filters["start_date"] <= r["date"] <= filters["end_date"]]
        
        return filtered
    
    def add_attendance_record(self, record: Dict) -> bool:
        """Add attendance record."""
        data = self._read_json(self.attendance_file, self.attendance_lock)
        records = data.get("records", [])
        
        # Generate ID
        record["id"] = max([r.get("id", 0) for r in records], default=0) + 1
        record["created_at"] = datetime.now().isoformat()
        
        records.append(record)
        data["records"] = records
        
        return self._write_json(self.attendance_file, data, self.attendance_lock)
    
    def get_statistics(self, date: Optional[str] = None) -> Dict:
        """Get attendance statistics."""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        records = self.get_attendance_records({"date": date})
        
        if not records:
            return {
                'date': date,
                'total_records': 0,
                'message': 'No attendance data for this date'
            }
        
        unique_people = list(set(r["name"] for r in records))
        check_ins = sum(1 for r in records if r["check_type"] == "IN")
        check_outs = sum(1 for r in records if r["check_type"] == "OUT")
        
        confidences = [r["confidence"] for r in records]
        qualities = [r["quality"] for r in records]
        
        times = [r["time"] for r in records]
        
        return {
            'date': date,
            'total_records': len(records),
            'unique_people': len(unique_people),
            'people': unique_people,
            'check_ins': check_ins,
            'check_outs': check_outs,
            'avg_confidence': round(sum(confidences) / len(confidences), 2) if confidences else 0,
            'avg_quality': round(sum(qualities) / len(qualities), 3) if qualities else 0,
            'first_check_in': min(times) if times else None,
            'last_check_out': max(times) if times else None
        }

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

# ============================================================================
# ATTENDANCE MANAGER CLASS
# ============================================================================

class AttendanceManager:
    """Core attendance management system with JSON storage."""
    
    def __init__(
        self,
        model_path: str = "models/combined_model.yml",
        data_dir: str = "attendance_data",
        security_level: str = "balanced",
        duplicate_threshold_minutes: int = 5,
        enable_antispoofing: bool = True
    ):
        """Initialize attendance management system."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        for subdir in ['logs', 'reports', 'exports', 'backups']:
            (self.data_dir / subdir).mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.storage = JSONStorage(data_dir)
        self.security_level = security_level
        self.duplicate_threshold = timedelta(minutes=duplicate_threshold_minutes)
        
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
            self.logger.warning(f"   ⚠️ No model found at: {model_path}")
        
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
                self.logger.warning(f"   ⚠️ Anti-spoofing initialization failed: {e}")
        
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
    
    def _load_today_records(self) -> None:
        """Load today's attendance records from JSON storage."""
        today = datetime.now().strftime('%Y-%m-%d')
        records = self.storage.get_attendance_records({"date": today})
        
        for record_dict in records:
            try:
                timestamp = datetime.fromisoformat(record_dict['timestamp'])
                record = AttendanceRecord(
                    user_id=record_dict['user_id'],
                    name=record_dict['name'],
                    timestamp=timestamp,
                    confidence=record_dict['confidence'],
                    quality=record_dict['quality'],
                    is_live=record_dict['is_live'],
                    check_type=record_dict['check_type']
                )
                self.attendance_records.append(record)
                
                if record_dict['check_type'] == "IN":
                    self.last_check_in[record_dict['name']] = timestamp
                else:
                    self.last_check_out[record_dict['name']] = timestamp
            except Exception as e:
                self.logger.warning(f"Error loading record: {e}")
        
        if self.attendance_records:
            self.logger.info(f"   📋 Loaded {len(self.attendance_records)} records from today")
    
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
        """Record attendance to JSON storage and memory."""
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
            
            # Save to JSON storage
            self.storage.add_attendance_record(record.to_dict())
            
            # Backup to CSV
            self._save_to_csv(record)
            
            self.logger.info(f"✅ {result.name} - {check_type} at {now.strftime('%H:%M:%S')}")
            
            return record
            
        except Exception as e:
            self.logger.error(f"Recording error: {e}", exc_info=True)
            raise
    
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

# ============================================================================
# AUTHENTICATION DECORATORS
# ============================================================================

def token_required(f):
    """Decorator to require valid JWT token."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Get token from header
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(" ")[1]  # Bearer <token>
            except IndexError:
                return jsonify({'error': 'Invalid token format'}), 401
        
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
        
        try:
            # Decode token
            data = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            current_user = storage.get_user_by_id(data['user_id'])
            
            if not current_user or not current_user.get('active', False):
                return jsonify({'error': 'Invalid or inactive user'}), 401
            
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(current_user, *args, **kwargs)
    
    return decorated


def admin_required(f):
    """Decorator to require admin role."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(" ")[1]
            except IndexError:
                return jsonify({'error': 'Invalid token format'}), 401
        
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
        
        try:
            data = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            current_user = storage.get_user_by_id(data['user_id'])
            
            if not current_user or not current_user.get('active', False):
                return jsonify({'error': 'Invalid or inactive user'}), 401
            
            if current_user.get('role') != 'admin':
                return jsonify({'error': 'Admin access required'}), 403
            
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(current_user, *args, **kwargs)
    
    return decorated

# ============================================================================
# FLASK APP INITIALIZATION
# ============================================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept"],
        "expose_headers": ["Content-Disposition", "Content-Type"],
        "supports_credentials": True
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
storage = None
camera = None
current_mode: str = "IN"
camera_lock = Lock()
system_lock = Lock()
camera_active = False

# ============================================================================
# INITIALIZATION
# ============================================================================

def init_system(model_path: str = "models/combined_model.yml", security_level: str = "balanced"):
    """Initialize the attendance system."""
    global attendance_manager, storage
    
    with system_lock:
        if storage is None:
            storage = JSONStorage("attendance_data")
            logger.info("✅ JSON storage initialized")
        
        if attendance_manager is None:
            logger.info("🚀 Initializing Attendance System...")
            try:
                attendance_manager = AttendanceManager(
                    model_path=model_path,
                    data_dir="attendance_data",
                    security_level=security_level,
                    enable_antispoofing=True
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
# AUTHENTICATION ROUTES
# ============================================================================

@app.route('/api/auth/login', methods=['POST'])
def login():
    """User login endpoint."""
    data = request.json or {}
    
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
    
    user = storage.get_user_by_username(username)
    
    if not user:
        return jsonify({'error': 'Invalid credentials'}), 401
    
    if not user.get('active', False):
        return jsonify({'error': 'Account is inactive'}), 401
    
    if not check_password_hash(user['password'], password):
        return jsonify({'error': 'Invalid credentials'}), 401
    
    # Generate JWT token
    token = jwt.encode({
        'user_id': user['id'],
        'username': user['username'],
        'role': user['role'],
        'exp': datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    }, SECRET_KEY, algorithm="HS256")
    
    logger.info(f"✅ User logged in: {username} ({user['role']})")
    
    return jsonify({
        'success': True,
        'token': token,
        'user': {
            'id': user['id'],
            'username': user['username'],
            'name': user['name'],
            'role': user['role'],
            'email': user['email']
        }
    })


@app.route('/api/auth/register', methods=['POST'])
@admin_required
def register(current_user):
    """Register new user (admin only)."""
    data = request.json or {}
    
    required_fields = ['username', 'password', 'name']
    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Missing required fields'}), 400
    
    # Check if username exists
    if storage.get_user_by_username(data['username']):
        return jsonify({'error': 'Username already exists'}), 400
    
    try:
        new_user = storage.create_user({
            'username': data['username'],
            'password': data['password'],
            'name': data['name'],
            'email': data.get('email', ''),
            'role': data.get('role', 'user')
        })
        
        logger.info(f"✅ New user created: {data['username']} by {current_user['username']}")
        
        return jsonify({
            'success': True,
            'user': {
                'id': new_user['id'],
                'username': new_user['username'],
                'name': new_user['name'],
                'role': new_user['role'],
                'email': new_user['email']
            }
        })
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/auth/me', methods=['GET'])
@token_required
def get_current_user(current_user):
    """Get current user info."""
    return jsonify({
        'user': {
            'id': current_user['id'],
            'username': current_user['username'],
            'name': current_user['name'],
            'role': current_user['role'],
            'email': current_user['email']
        }
    })


@app.route('/api/auth/change-password', methods=['POST'])
@token_required
def change_password(current_user):
    """Change user password."""
    data = request.json or {}
    
    old_password = data.get('old_password')
    new_password = data.get('new_password')
    
    if not old_password or not new_password:
        return jsonify({'error': 'Old and new passwords required'}), 400
    
    if not check_password_hash(current_user['password'], old_password):
        return jsonify({'error': 'Invalid old password'}), 401
    
    if len(new_password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400
    
    try:
        storage.update_user(current_user['id'], {'password': new_password})
        logger.info(f"✅ Password changed for user: {current_user['username']}")
        return jsonify({'success': True, 'message': 'Password changed successfully'})
    except Exception as e:
        logger.error(f"Password change error: {e}")
        return jsonify({'error': str(e)}), 500

# ============================================================================
# VIDEO STREAMING (Token Required)
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
                socketio.emit('new_record', record.to_dict())
                socketio.emit('stats_update', get_quick_stats())
            
            h, w = annotated.shape[:2]
            
            mode_color = (0, 255, 0) if current_mode == "IN" else (255, 140, 0)
            cv2.rectangle(annotated, (10, 10), (250, 80), mode_color, -1)
            cv2.putText(annotated, f"Mode: CHECK-{current_mode}",
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(annotated, f"Records: {len(attendance_manager.attendance_records)}",
                       (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
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
    """Health check endpoint (no auth required)."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'system_initialized': attendance_manager is not None,
        'camera_active': camera_active
    })


@app.route('/api/video/feed')
@token_required
def video_feed(current_user):
    """Video streaming endpoint (requires authentication)."""
    if not attendance_manager:
        return jsonify({'error': 'System not initialized'}), 503
    
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/mode', methods=['GET'])
@token_required
def get_mode(current_user):
    """Get current mode (users can view)."""
    return jsonify({'mode': current_mode})


@app.route('/api/mode', methods=['POST'])
@token_required
def set_mode(current_user):
    """Set attendance mode (users can change for their own check-in/out)."""
    global current_mode
    
    data = request.json or {}
    new_mode = data.get('mode', '').upper()
    
    if new_mode not in ['IN', 'OUT']:
        return jsonify({'error': 'Invalid mode'}), 400
    
    current_mode = new_mode
    socketio.emit('mode_changed', {'mode': current_mode})
    
    logger.info(f"Mode changed to: {current_mode} by {current_user['username']}")
    return jsonify({'success': True, 'mode': current_mode})


@app.route('/api/statistics', methods=['GET'])
@admin_required
def get_statistics(current_user):
    """Get attendance statistics (admin only)."""
    if not storage:
        return jsonify({'error': 'System not initialized'}), 503
    
    date = request.args.get('date')
    
    try:
        stats = storage.get_statistics(date)
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Statistics error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/statistics/quick', methods=['GET'])
@token_required
def quick_statistics(current_user):
    """Get quick statistics (authenticated users can view)."""
    if not attendance_manager:
        return jsonify({'error': 'System not initialized'}), 503
    
    return jsonify(get_quick_stats())


@app.route('/api/records/today', methods=['GET'])
@admin_required
def get_today_records(current_user):
    """Get today's records (admin only)."""
    if not storage:
        return jsonify({'error': 'System not initialized'}), 503
    
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        records = storage.get_attendance_records({'date': today})
        return jsonify({'records': records, 'count': len(records)})
    except Exception as e:
        logger.error(f"Error fetching records: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/records/my', methods=['GET'])
@token_required
def get_my_records(current_user):
    """Get current user's records."""
    if not storage:
        return jsonify({'error': 'System not initialized'}), 503
    
    try:
        # Get records for current user
        days = int(request.args.get('days', 7))
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        all_records = storage.get_attendance_records({
            'start_date': start_date,
            'end_date': end_date
        })
        
        # Filter by user name (you might want to match by user_id if you have it)
        my_records = [r for r in all_records if r['name'] == current_user['name']]
        
        return jsonify({'records': my_records, 'count': len(my_records)})
    except Exception as e:
        logger.error(f"Error fetching user records: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/records/range', methods=['GET'])
@admin_required
def get_date_range_records(current_user):
    """Get records for date range (admin only)."""
    if not storage:
        return jsonify({'error': 'System not initialized'}), 503
    
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    if not start_date or not end_date:
        return jsonify({'error': 'start_date and end_date required'}), 400
    
    try:
        records = storage.get_attendance_records({
            'start_date': start_date,
            'end_date': end_date
        })
        return jsonify({'records': records, 'count': len(records)})
    except Exception as e:
        logger.error(f"Date range query error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/users', methods=['GET'])
@admin_required
def get_users(current_user):
    """Get all users (admin only)."""
    if not storage:
        return jsonify({'error': 'System not initialized'}), 503
    
    try:
        users = storage.get_users()
        # Remove password hashes from response
        safe_users = [{k: v for k, v in u.items() if k != 'password'} for u in users]
        return jsonify({'users': safe_users, 'count': len(safe_users)})
    except Exception as e:
        logger.error(f"Users query error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/users/<int:user_id>', methods=['PUT'])
@admin_required
def update_user(current_user, user_id):
    """Update user (admin only)."""
    if not storage:
        return jsonify({'error': 'System not initialized'}), 503
    
    data = request.json or {}
    
    try:
        success = storage.update_user(user_id, data)
        if success:
            return jsonify({'success': True, 'message': 'User updated'})
        else:
            return jsonify({'error': 'User not found'}), 404
    except Exception as e:
        logger.error(f"User update error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/users/<int:user_id>', methods=['DELETE'])
@admin_required
def delete_user(current_user, user_id):
    """Delete user (admin only)."""
    if not storage:
        return jsonify({'error': 'System not initialized'}), 503
    
    if user_id == current_user['id']:
        return jsonify({'error': 'Cannot delete your own account'}), 400
    
    try:
        success = storage.delete_user(user_id)
        if success:
            return jsonify({'success': True, 'message': 'User deleted'})
        else:
            return jsonify({'error': 'User not found'}), 404
    except Exception as e:
        logger.error(f"User deletion error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/export', methods=['GET'])
@admin_required
def export_data(current_user):
    """Export attendance data (admin only)."""
    if not storage:
        return jsonify({'error': 'System not initialized'}), 503
    
    date = request.args.get('date')
    format_type = request.args.get('format', 'csv')
    
    if format_type not in ['csv', 'json']:
        return jsonify({'error': 'Invalid format. Use: csv or json'}), 400
    
    try:
        if date:
            records = storage.get_attendance_records({'date': date})
        else:
            date = datetime.now().strftime('%Y-%m-%d')
            records = storage.get_attendance_records({'date': date})
        
        if not records:
            return jsonify({'error': f'No attendance data for {date}'}), 404
        
        export_dir = Path("attendance_data/exports")
        export_dir.mkdir(exist_ok=True)
        
        if format_type == 'csv':
            output_file = export_dir / f"attendance_{date}.csv"
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                if records:
                    writer = csv.DictWriter(f, fieldnames=records[0].keys())
                    writer.writeheader()
                    writer.writerows(records)
            
            mime_type = 'text/csv'
            
        else:  # json
            output_file = export_dir / f"attendance_{date}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
            
            mime_type = 'application/json'
        
        logger.info(f"📊 Exported {len(records)} records to: {output_file}")
        
        return send_file(
            output_file,
            mimetype=mime_type,
            as_attachment=True,
            download_name=output_file.name
        )
        
    except Exception as e:
        logger.error(f"Export error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/backup', methods=['POST'])
@admin_required
def create_backup(current_user):
    """Create backup of all JSON files (admin only)."""
    if not storage:
        return jsonify({'error': 'System not initialized'}), 503
    
    try:
        backup_dir = Path("attendance_data/backups")
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"backup_{timestamp}"
        backup_path = backup_dir / backup_name
        backup_path.mkdir(exist_ok=True)
        
        # Copy all JSON files
        data_dir = Path("attendance_data")
        for json_file in ['users.json', 'attendance.json', 'settings.json']:
            src = data_dir / json_file
            if src.exists():
                shutil.copy2(src, backup_path / json_file)
        
        logger.info(f"💾 Backup created: {backup_path}")
        
        return jsonify({
            'success': True,
            'path': str(backup_path),
            'message': f'Backup created successfully',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Backup error: {e}", exc_info=True)
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
    
    parser = argparse.ArgumentParser(description='Attendance System Flask Backend with Auth')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--model', default='models/combined_model.yml', help='Model path')
    parser.add_argument('--security', default='balanced', choices=['lenient', 'balanced', 'strict'])
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🚀 ATTENDANCE SYSTEM - FLASK BACKEND API WITH AUTHENTICATION")
    print("="*70)
    print(f"   URL: http://{args.host}:{args.port}")
    print(f"   Security: {args.security.upper()}")
    print(f"   Storage: JSON Files")
    print("="*70)
    print("\n   Default Credentials:")
    print("   Admin - username: admin, password: admin123")
    print("   User  - username: user,  password: user123")
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