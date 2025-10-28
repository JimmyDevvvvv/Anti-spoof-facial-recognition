"""
Attendance Database Module

Manages attendance records with SQLite database.
Provides logging, querying, and reporting functionality.
"""

import sqlite3
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Optional


class AttendanceDatabase:
    """
    Attendance management with SQLite database.
    """
    
    def __init__(self, db_path: str = "attendance.db"):
        """
        Initialize attendance database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # People table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS people (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    label_id INTEGER UNIQUE NOT NULL,
                    email TEXT,
                    department TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Attendance records table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    confidence REAL,
                    status TEXT DEFAULT 'present',
                    FOREIGN KEY (person_id) REFERENCES people (id)
                )
            """)
            
            # Daily summary table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    person_id INTEGER NOT NULL,
                    check_in_time TIMESTAMP,
                    check_out_time TIMESTAMP,
                    total_hours REAL,
                    UNIQUE(date, person_id),
                    FOREIGN KEY (person_id) REFERENCES people (id)
                )
            """)
            
            conn.commit()
    
    def add_person(
        self,
        name: str,
        label_id: int,
        email: Optional[str] = None,
        department: Optional[str] = None
    ) -> int:
        """
        Add a person to the database.
        
        Args:
            name: Person's name
            label_id: Recognition model label ID
            email: Email address (optional)
            department: Department/class (optional)
            
        Returns:
            Person ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO people (name, label_id, email, department)
                VALUES (?, ?, ?, ?)
                """,
                (name, label_id, email, department)
            )
            conn.commit()
            return cursor.lastrowid
    
    def log_attendance(
        self,
        person_name: str,
        confidence: float,
        status: str = "present"
    ) -> bool:
        """
        Log attendance for a person.
        
        Args:
            person_name: Name of the person
            confidence: Recognition confidence score
            status: Attendance status (present, late, etc.)
            
        Returns:
            True if logged successfully
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get person ID
            cursor.execute(
                "SELECT id FROM people WHERE name = ?",
                (person_name,)
            )
            result = cursor.fetchone()
            
            if not result:
                print(f"Person not found in database: {person_name}")
                return False
            
            person_id = result[0]
            
            # Check if already logged today
            today = date.today()
            cursor.execute(
                """
                SELECT COUNT(*) FROM attendance
                WHERE person_id = ? AND DATE(timestamp) = ?
                """,
                (person_id, today)
            )
            
            count = cursor.fetchone()[0]
            
            if count > 0:
                print(f"{person_name} already logged today")
                return False
            
            # Log attendance
            cursor.execute(
                """
                INSERT INTO attendance (person_id, confidence, status)
                VALUES (?, ?, ?)
                """,
                (person_id, confidence, status)
            )
            
            # Update daily summary
            cursor.execute(
                """
                INSERT OR REPLACE INTO daily_summary 
                (date, person_id, check_in_time)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                """,
                (today, person_id)
            )
            
            conn.commit()
            return True
    
    def get_today_attendance(self) -> List[Dict]:
        """
        Get today's attendance records.
        
        Returns:
            List of attendance records
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            today = date.today()
            
            cursor.execute(
                """
                SELECT p.name, a.timestamp, a.confidence, a.status
                FROM attendance a
                JOIN people p ON a.person_id = p.id
                WHERE DATE(a.timestamp) = ?
                ORDER BY a.timestamp DESC
                """,
                (today,)
            )
            
            records = []
            for row in cursor.fetchall():
                records.append({
                    'name': row[0],
                    'timestamp': row[1],
                    'confidence': row[2],
                    'status': row[3]
                })
            
            return records
    
    def get_attendance_report(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> List[Dict]:
        """
        Generate attendance report for date range.
        
        Args:
            start_date: Start date (default: today)
            end_date: End date (default: today)
            
        Returns:
            List of attendance records
        """
        if start_date is None:
            start_date = date.today()
        if end_date is None:
            end_date = date.today()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                """
                SELECT p.name, p.department, DATE(a.timestamp) as date,
                       a.timestamp, a.status
                FROM attendance a
                JOIN people p ON a.person_id = p.id
                WHERE DATE(a.timestamp) BETWEEN ? AND ?
                ORDER BY a.timestamp DESC
                """,
                (start_date, end_date)
            )
            
            records = []
            for row in cursor.fetchall():
                records.append({
                    'name': row[0],
                    'department': row[1],
                    'date': row[2],
                    'timestamp': row[3],
                    'status': row[4]
                })
            
            return records
    
    def get_absentees(self, target_date: Optional[date] = None) -> List[str]:
        """
        Get list of people who were absent on a given date.
        
        Args:
            target_date: Date to check (default: today)
            
        Returns:
            List of names who were absent
        """
        if target_date is None:
            target_date = date.today()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                """
                SELECT p.name
                FROM people p
                WHERE p.id NOT IN (
                    SELECT person_id FROM attendance
                    WHERE DATE(timestamp) = ?
                )
                ORDER BY p.name
                """,
                (target_date,)
            )
            
            return [row[0] for row in cursor.fetchall()]
    
    def export_to_csv(self, output_path: str, start_date: date, end_date: date):
        """
        Export attendance records to CSV.
        
        Args:
            output_path: Path to save CSV file
            start_date: Start date
            end_date: End date
        """
        import csv
        
        records = self.get_attendance_report(start_date, end_date)
        
        with open(output_path, 'w', newline='') as f:
            if records:
                writer = csv.DictWriter(f, fieldnames=records[0].keys())
                writer.writeheader()
                writer.writerows(records)
        
        print(f"Exported {len(records)} records to {output_path}")
    
    def get_all_people(self) -> List[Dict]:
        """
        Get all people from database.
        
        Returns:
            List of people with their details
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, name, label_id, email, department, created_at
                FROM people
                ORDER BY name
                """
            )
            
            people = []
            for row in cursor.fetchall():
                people.append({
                    'id': row[0],
                    'name': row[1],
                    'label_id': row[2],
                    'email': row[3],
                    'department': row[4],
                    'created_at': row[5]
                })
            
            return people
    
    def get_person_by_id(self, person_id: int) -> Optional[Dict]:
        """
        Get person details by ID.
        
        Args:
            person_id: Person ID
            
        Returns:
            Person details or None
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, name, label_id, email, department, created_at
                FROM people
                WHERE id = ?
                """,
                (person_id,)
            )
            
            row = cursor.fetchone()
            if row:
                return {
                    'id': row[0],
                    'name': row[1],
                    'label_id': row[2],
                    'email': row[3],
                    'department': row[4],
                    'created_at': row[5]
                }
            return None
    
    def update_person(
        self,
        person_id: int,
        name: Optional[str] = None,
        email: Optional[str] = None,
        department: Optional[str] = None
    ) -> bool:
        """
        Update person details.
        
        Args:
            person_id: Person ID
            name: New name (optional)
            email: New email (optional)
            department: New department (optional)
            
        Returns:
            True if updated successfully
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            updates = []
            params = []
            
            if name is not None:
                updates.append("name = ?")
                params.append(name)
            if email is not None:
                updates.append("email = ?")
                params.append(email)
            if department is not None:
                updates.append("department = ?")
                params.append(department)
            
            if not updates:
                return False
            
            params.append(person_id)
            query = f"UPDATE people SET {', '.join(updates)} WHERE id = ?"
            
            cursor.execute(query, params)
            conn.commit()
            return cursor.rowcount > 0
    
    def delete_person(self, person_id: int) -> bool:
        """
        Delete person and their attendance records.
        
        Args:
            person_id: Person ID
            
        Returns:
            True if deleted successfully
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Delete attendance records
            cursor.execute("DELETE FROM attendance WHERE person_id = ?", (person_id,))
            cursor.execute("DELETE FROM daily_summary WHERE person_id = ?", (person_id,))
            
            # Delete person
            cursor.execute("DELETE FROM people WHERE id = ?", (person_id,))
            
            conn.commit()
            return cursor.rowcount > 0
    
    def delete_attendance_record(self, record_id: int) -> bool:
        """
        Delete an attendance record.
        
        Args:
            record_id: Attendance record ID
            
        Returns:
            True if deleted successfully
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM attendance WHERE id = ?", (record_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    def get_person_attendance_history(
        self,
        person_id: int,
        limit: int = 10
    ) -> List[Dict]:
        """
        Get attendance history for a person.
        
        Args:
            person_id: Person ID
            limit: Maximum number of records
            
        Returns:
            List of attendance records
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT timestamp, confidence, status
                FROM attendance
                WHERE person_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (person_id, limit)
            )
            
            records = []
            for row in cursor.fetchall():
                records.append({
                    'timestamp': row[0],
                    'confidence': row[1],
                    'status': row[2]
                })
            
            return records
    
    def get_settings(self) -> Dict:
        """
        Get system settings.
        
        Returns:
            Dictionary of settings
        """
        # For now, return default settings
        # In production, these would be stored in a settings table
        return {
            'system_name': 'Attendance Management System',
            'confidence_threshold': 50.0,
            'detection_mode': 'highacc',
            'working_days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
            'enable_notifications': False,
            'admin_email': ''
        }
    
    def update_settings(self, settings: Dict) -> bool:
        """
        Update system settings.
        
        Args:
            settings: Dictionary of settings to update
            
        Returns:
            True if updated successfully
        """
        # For now, just return True
        # In production, these would be stored in a settings table
        return True


# Example usage
if __name__ == "__main__":
    # Initialize database
    db = AttendanceDatabase("attendance.db")
    
    # Add people (do this once per person)
    # db.add_person("John Doe", label_id=0, email="john@example.com", department="Engineering")
    # db.add_person("Jane Smith", label_id=1, email="jane@example.com", department="HR")
    
    # Log attendance (called from recognition system)
    # db.log_attendance("John Doe", confidence=45.2, status="present")
    
    # Get today's attendance
    print("Today's Attendance:")
    for record in db.get_today_attendance():
        print(f"  {record['name']}: {record['timestamp']} (confidence: {record['confidence']})")
    
    # Get absentees
    print("\nAbsentees:")
    for name in db.get_absentees():
        print(f"  {name}")
    
    # Export to CSV
    from datetime import date, timedelta
    end_date = date.today()
    start_date = end_date - timedelta(days=7)
    db.export_to_csv("attendance_report.csv", start_date, end_date)

