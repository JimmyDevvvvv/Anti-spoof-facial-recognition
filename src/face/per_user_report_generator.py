"""
Per-User Report Generator
=========================

Tracks and generates detailed reports for each user's recognition sessions.
Automatically logs every recognition attempt and generates comprehensive reports.

Features:
- Real-time per-user tracking
- Session-based statistics
- Automatic report generation
- Anti-spoofing tracking
- Quality metrics
- Export to JSON and text formats
"""

import json
import sqlite3
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


class PerUserReportGenerator:
    """
    Generates detailed per-user reports for face recognition sessions.
    Tracks every recognition attempt with comprehensive metrics.
    """
    
    def __init__(
        self,
        db_path: str = "user_reports.db",
        reports_dir: str = "reports",
        auto_save_interval: int = 50  # Auto-save after N attempts
    ):
        """
        Initialize the per-user report generator.
        
        Args:
            db_path: Path to SQLite database for persistent storage
            reports_dir: Directory to save report files
            auto_save_interval: Number of attempts before auto-saving report
        """
        self.db_path = db_path
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True)
        self.auto_save_interval = auto_save_interval
        
        # Session tracking
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_start = time.time()
        
        # Per-user statistics
        self.user_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'total_attempts': 0,
            'successful_recognitions': 0,
            'failed_attempts': 0,
            'confidences': [],
            'quality_scores': [],
            'processing_times': [],
            'anti_spoofing': {
                'total_checks': 0,
                'real_detections': 0,
                'spoof_detections': 0,
                'spoof_types': defaultdict(int)
            },
            'edge_cases': {
                'low_quality': 0,
                'low_confidence': 0,
                'high_confidence': 0
            },
            'timestamps': [],
            'session_durations': []
        })
        
        # Unknown/rejected tracking
        self.unknown_attempts = 0
        self.false_positives = []
        
        # Total statistics
        self.total_attempts = 0
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                start_time REAL,
                end_time REAL,
                total_attempts INTEGER,
                unknown_attempts INTEGER
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recognition_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp REAL,
                user_name TEXT,
                recognized BOOLEAN,
                confidence REAL,
                quality_score REAL,
                processing_time REAL,
                is_spoof BOOLEAN,
                spoof_type TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_statistics (
                session_id TEXT,
                user_name TEXT,
                total_attempts INTEGER,
                successful_recognitions INTEGER,
                avg_confidence REAL,
                avg_quality REAL,
                avg_processing_time REAL,
                anti_spoofing_checks INTEGER,
                spoofs_detected INTEGER,
                PRIMARY KEY (session_id, user_name),
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        
        conn.commit()
        conn.close()
        
        # Insert session record
        self._save_session_start()
    
    def _save_session_start(self):
        """Save session start to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO sessions (session_id, start_time, total_attempts, unknown_attempts)
            VALUES (?, ?, 0, 0)
        """, (self.session_id, self.session_start))
        conn.commit()
        conn.close()
    
    def log_recognition_attempt(
        self,
        user_name: Optional[str],
        recognized: bool,
        confidence: float,
        quality_score: float = 0.0,
        processing_time: float = 0.0,
        is_spoof: bool = False,
        spoof_type: Optional[str] = None
    ):
        """
        Log a single recognition attempt.
        
        Args:
            user_name: Name of recognized user (None if unknown)
            recognized: Whether recognition was successful
            confidence: Recognition confidence score
            quality_score: Image quality score
            processing_time: Processing time in seconds
            is_spoof: Whether anti-spoofing detected a spoof
            spoof_type: Type of spoof detected (if any)
        """
        self.total_attempts += 1
        timestamp = time.time()
        
        if user_name is None or not recognized:
            self.unknown_attempts += 1
            user_name = "Unknown"
        
        # Update user statistics
        stats = self.user_stats[user_name]
        stats['total_attempts'] += 1
        
        if recognized and user_name != "Unknown":
            stats['successful_recognitions'] += 1
            stats['confidences'].append(confidence)
            stats['quality_scores'].append(quality_score)
            stats['processing_times'].append(processing_time)
            stats['timestamps'].append(timestamp)
            
            # Edge case tracking
            if quality_score < 0.5:
                stats['edge_cases']['low_quality'] += 1
            if confidence > 50:
                stats['edge_cases']['low_confidence'] += 1
            elif confidence < 30:
                stats['edge_cases']['high_confidence'] += 1
        else:
            stats['failed_attempts'] += 1
        
        # Anti-spoofing tracking
        if is_spoof:
            stats['anti_spoofing']['total_checks'] += 1
            stats['anti_spoofing']['spoof_detections'] += 1
            if spoof_type:
                stats['anti_spoofing']['spoof_types'][spoof_type] += 1
        elif recognized:
            stats['anti_spoofing']['total_checks'] += 1
            stats['anti_spoofing']['real_detections'] += 1
        
        # Save to database
        self._save_attempt_to_db(
            timestamp, user_name, recognized, confidence,
            quality_score, processing_time, is_spoof, spoof_type
        )
        
        # Auto-save report periodically
        if self.total_attempts % self.auto_save_interval == 0:
            self.generate_report(auto_save=True)
    
    def _save_attempt_to_db(
        self,
        timestamp: float,
        user_name: str,
        recognized: bool,
        confidence: float,
        quality_score: float,
        processing_time: float,
        is_spoof: bool,
        spoof_type: Optional[str]
    ):
        """Save individual attempt to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO recognition_attempts 
            (session_id, timestamp, user_name, recognized, confidence, 
             quality_score, processing_time, is_spoof, spoof_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            self.session_id, timestamp, user_name, recognized, confidence,
            quality_score, processing_time, is_spoof, spoof_type
        ))
        conn.commit()
        conn.close()
    
    def generate_report(self, auto_save: bool = False) -> Dict[str, Any]:
        """
        Generate comprehensive per-user report.
        
        Args:
            auto_save: Whether this is an automatic periodic save
            
        Returns:
            Complete report dictionary
        """
        session_duration = time.time() - self.session_start
        
        # Build report structure
        report = {
            'session_info': {
                'session_id': self.session_id,
                'start_time': datetime.fromtimestamp(self.session_start).isoformat(),
                'duration_seconds': round(session_duration, 2),
                'total_attempts': self.total_attempts,
                'unknown_attempts': self.unknown_attempts
            },
            'per_user_details': {},
            'summary': {
                'total_users': len([u for u in self.user_stats.keys() if u != "Unknown"]),
                'overall_accuracy': 0.0,
                'total_spoofs_detected': 0
            }
        }
        
        # Calculate per-user statistics
        total_successful = 0
        total_user_attempts = 0
        
        for user_name, stats in self.user_stats.items():
            if user_name == "Unknown":
                continue
            
            attempts = stats['total_attempts']
            successes = stats['successful_recognitions']
            
            if attempts == 0:
                continue
            
            total_successful += successes
            total_user_attempts += attempts
            
            # Calculate metrics
            recognition_rate = (successes / attempts * 100) if attempts > 0 else 0
            avg_confidence = (sum(stats['confidences']) / len(stats['confidences'])) if stats['confidences'] else 0
            avg_quality = (sum(stats['quality_scores']) / len(stats['quality_scores'])) if stats['quality_scores'] else 0
            avg_processing = (sum(stats['processing_times']) / len(stats['processing_times'])) if stats['processing_times'] else 0
            
            # Anti-spoofing metrics
            anti_spoof = stats['anti_spoofing']
            spoof_detection_rate = (
                (anti_spoof['spoof_detections'] / anti_spoof['total_checks'] * 100)
                if anti_spoof['total_checks'] > 0 else 0
            )
            
            report['per_user_details'][user_name] = {
                'total_attempts': attempts,
                'successful_recognitions': successes,
                'failed_attempts': stats['failed_attempts'],
                'recognition_rate': round(recognition_rate, 2),
                'average_confidence': round(avg_confidence, 2),
                'confidence_range': {
                    'min': round(min(stats['confidences']), 2) if stats['confidences'] else 0,
                    'max': round(max(stats['confidences']), 2) if stats['confidences'] else 0
                },
                'average_quality_score': round(avg_quality, 3),
                'average_processing_time_ms': round(avg_processing * 1000, 2),
                'anti_spoofing': {
                    'total_checks': anti_spoof['total_checks'],
                    'real_detections': anti_spoof['real_detections'],
                    'spoof_detections': anti_spoof['spoof_detections'],
                    'detection_rate': round(spoof_detection_rate, 2),
                    'spoof_types': dict(anti_spoof['spoof_types'])
                },
                'edge_cases': {
                    'low_quality_detections': stats['edge_cases']['low_quality'],
                    'low_confidence_cases': stats['edge_cases']['low_confidence'],
                    'high_confidence_cases': stats['edge_cases']['high_confidence']
                },
                'sample_confidences': [round(c, 2) for c in stats['confidences'][:10]]
            }
        
        # Calculate summary metrics
        overall_accuracy = (total_successful / total_user_attempts * 100) if total_user_attempts > 0 else 0
        report['summary']['overall_accuracy'] = round(overall_accuracy, 2)
        report['summary']['total_spoofs_detected'] = sum(
            s['anti_spoofing']['spoof_detections'] 
            for s in self.user_stats.values()
        )
        
        # Save reports to files
        if not auto_save or self.total_attempts >= self.auto_save_interval:
            self._save_report_to_files(report, auto_save)
        
        # Update database with final statistics
        self._update_user_statistics()
        
        return report
    
    def _save_report_to_files(self, report: Dict[str, Any], auto_save: bool = False):
        """Save report to JSON and text files."""
        suffix = "_auto" if auto_save else ""
        
        # JSON report
        json_path = self.reports_dir / f"user_report_{self.session_id}{suffix}.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Text report
        txt_path = self.reports_dir / f"user_report_{self.session_id}{suffix}.txt"
        with open(txt_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DETAILED PER-USER RECOGNITION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Session info
            f.write("SESSION INFORMATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Session ID: {report['session_info']['session_id']}\n")
            f.write(f"Start Time: {report['session_info']['start_time']}\n")
            f.write(f"Duration: {report['session_info']['duration_seconds']}s\n")
            f.write(f"Total Attempts: {report['session_info']['total_attempts']}\n")
            f.write(f"Unknown Attempts: {report['session_info']['unknown_attempts']}\n\n")
            
            # Summary
            f.write("OVERALL SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Users Recognized: {report['summary']['total_users']}\n")
            f.write(f"Overall Accuracy: {report['summary']['overall_accuracy']}%\n")
            f.write(f"Total Spoofs Detected: {report['summary']['total_spoofs_detected']}\n\n")
            
            # Per-user details
            f.write("=" * 80 + "\n")
            f.write("PER-USER DETAILED RESULTS\n")
            f.write("=" * 80 + "\n\n")
            
            for user_name, details in report['per_user_details'].items():
                f.write(f"\nUser: {user_name}\n")
                f.write("-" * 80 + "\n")
                f.write(f"  Total Attempts: {details['total_attempts']}\n")
                f.write(f"  Successful Recognitions: {details['successful_recognitions']}\n")
                f.write(f"  Failed Attempts: {details['failed_attempts']}\n")
                f.write(f"  Recognition Rate: {details['recognition_rate']}%\n\n")
                
                f.write(f"  Average Confidence: {details['average_confidence']}\n")
                f.write(f"  Confidence Range: {details['confidence_range']['min']} - {details['confidence_range']['max']}\n")
                f.write(f"  Average Quality Score: {details['average_quality_score']}\n")
                f.write(f"  Average Processing Time: {details['average_processing_time_ms']}ms\n\n")
                
                # Anti-spoofing
                anti_spoof = details['anti_spoofing']
                f.write(f"  Anti-Spoofing Checks:\n")
                f.write(f"    Total Checks: {anti_spoof['total_checks']}\n")
                f.write(f"    Real Detections: {anti_spoof['real_detections']}\n")
                f.write(f"    Spoof Detections: {anti_spoof['spoof_detections']}\n")
                f.write(f"    Detection Rate: {anti_spoof['detection_rate']}%\n")
                if anti_spoof['spoof_types']:
                    f.write(f"    Spoof Types: {', '.join(f'{k}({v})' for k, v in anti_spoof['spoof_types'].items())}\n")
                f.write("\n")
                
                # Edge cases
                edge = details['edge_cases']
                f.write(f"  Edge Case Performance:\n")
                f.write(f"    Low Quality Detections: {edge['low_quality_detections']}\n")
                f.write(f"    Low Confidence Cases: {edge['low_confidence_cases']}\n")
                f.write(f"    High Confidence Cases: {edge['high_confidence_cases']}\n\n")
                
                # Sample confidences
                if details['sample_confidences']:
                    f.write(f"  Sample Confidences (First 10): {', '.join(map(str, details['sample_confidences']))}\n")
                
                f.write("\n")
        
        print(f"\n{'[AUTO-SAVE]' if auto_save else '[REPORT GENERATED]'}")
        print(f"  JSON: {json_path}")
        print(f"  Text: {txt_path}")
    
    def _update_user_statistics(self):
        """Update user statistics in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for user_name, stats in self.user_stats.items():
            if user_name == "Unknown":
                continue
            
            avg_confidence = (sum(stats['confidences']) / len(stats['confidences'])) if stats['confidences'] else 0
            avg_quality = (sum(stats['quality_scores']) / len(stats['quality_scores'])) if stats['quality_scores'] else 0
            avg_processing = (sum(stats['processing_times']) / len(stats['processing_times'])) if stats['processing_times'] else 0
            
            cursor.execute("""
                INSERT OR REPLACE INTO user_statistics
                (session_id, user_name, total_attempts, successful_recognitions,
                 avg_confidence, avg_quality, avg_processing_time,
                 anti_spoofing_checks, spoofs_detected)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.session_id,
                user_name,
                stats['total_attempts'],
                stats['successful_recognitions'],
                avg_confidence,
                avg_quality,
                avg_processing,
                stats['anti_spoofing']['total_checks'],
                stats['anti_spoofing']['spoof_detections']
            ))
        
        # Update session end time
        cursor.execute("""
            UPDATE sessions 
            SET end_time = ?, total_attempts = ?, unknown_attempts = ?
            WHERE session_id = ?
        """, (time.time(), self.total_attempts, self.unknown_attempts, self.session_id))
        
        conn.commit()
        conn.close()
    
    def finalize_session(self) -> Dict[str, Any]:
        """
        Finalize the session and generate final report.
        
        Returns:
            Final complete report
        """
        print("\n" + "=" * 80)
        print("FINALIZING SESSION AND GENERATING FINAL REPORT")
        print("=" * 80)
        
        report = self.generate_report(auto_save=False)
        
        print("\nSession Statistics:")
        print(f"  Total Attempts: {self.total_attempts}")
        print(f"  Total Users: {report['summary']['total_users']}")
        print(f"  Overall Accuracy: {report['summary']['overall_accuracy']}%")
        print(f"  Total Spoofs Detected: {report['summary']['total_spoofs_detected']}")
        print("\nDetailed per-user report saved to 'reports' directory.")
        
        return report


if __name__ == "__main__":
    # Example usage
    print("Per-User Report Generator - Test")
    print("=" * 80)
    
    generator = PerUserReportGenerator()
    
    # Simulate some recognition attempts
    generator.log_recognition_attempt("OMAR", True, 28.5, 0.85, 0.095)
    generator.log_recognition_attempt("OMAR", True, 31.2, 0.78, 0.102)
    generator.log_recognition_attempt("NOUR", True, 29.8, 0.92, 0.088)
    generator.log_recognition_attempt(None, False, 55.0, 0.45, 0.105)  # Unknown
    generator.log_recognition_attempt("OMAR", True, 33.1, 0.81, 0.098)
    generator.log_recognition_attempt("NOUR", False, 52.0, 0.50, 0.115)  # Failed
    generator.log_recognition_attempt("OMAR", True, 27.9, 0.88, 0.091, is_spoof=True, spoof_type="photo")
    
    # Generate final report
    report = generator.finalize_session()
    
    print("\nTest completed successfully!")
