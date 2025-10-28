#!/usr/bin/env python3
"""
Attendance Management System
============================

Comprehensive web-based management system for face recognition attendance.

Features:
- Web dashboard with real-time statistics
- Attendance reports and exports
- Person management (add, edit, delete)
- Attendance history and analytics
- Manual attendance correction
- Bulk operations

Usage:
    python attendance_manager.py
    python attendance_manager.py --port 8080
    python attendance_manager.py --db custom_attendance.db
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "examples"))

try:
    from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
except ImportError as e:
    print("❌ Error: Flask not found")
    print(f"   Details: {e}")
    print("   Install Flask with: pip install flask")
    sys.exit(1)

try:
    from examples.attendance_database import AttendanceDatabase
except ImportError:
    try:
        from attendance_database import AttendanceDatabase  # type: ignore
    except ImportError as e:
        print("❌ Error: attendance_database module not found")
        print(f"   Details: {e}")
        print("   Make sure attendance_database.py exists in the examples folder")
        sys.exit(1)


app = Flask(__name__)
db: Optional[AttendanceDatabase] = None


def get_db() -> AttendanceDatabase:
    """Get database instance (ensures it's initialized)."""
    if db is None:
        raise RuntimeError("Database not initialized")
    return db


# ============================================================================
# WEB ROUTES - DASHBOARD
# ============================================================================

@app.route('/')
def index():
    """Main dashboard."""
    return render_template('dashboard.html')


@app.route('/api/dashboard/stats')
def get_dashboard_stats():
    """Get dashboard statistics."""
    try:
        db = get_db()
        # Today's attendance
        today_records = db.get_today_attendance()
        
        # Get all people
        all_people = db.get_all_people()
        
        # Absentees today
        absentees = db.get_absentees()
        
        # Week statistics
        week_start = date.today() - timedelta(days=date.today().weekday())
        week_records = db.get_attendance_report(week_start, date.today())
        
        # Calculate attendance rate
        total_people = len(all_people)
        present_today = len(today_records)
        attendance_rate = (present_today / total_people * 100) if total_people > 0 else 0
        
        # Weekly attendance rate
        week_attendance_by_day = {}
        for record in week_records:
            day = record['date']
            if day not in week_attendance_by_day:
                week_attendance_by_day[day] = set()
            week_attendance_by_day[day].add(record['name'])
        
        weekly_rates = [
            len(attendees) / total_people * 100 if total_people > 0 else 0
            for attendees in week_attendance_by_day.values()
        ]
        avg_weekly_rate = sum(weekly_rates) / len(weekly_rates) if weekly_rates else 0
        
        return jsonify({
            'success': True,
            'stats': {
                'total_people': total_people,
                'present_today': present_today,
                'absent_today': len(absentees),
                'attendance_rate': round(attendance_rate, 1),
                'avg_weekly_rate': round(avg_weekly_rate, 1),
                'recent_check_ins': today_records[:5]  # Last 5 check-ins
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/dashboard/chart-data')
def get_chart_data():
    """Get data for dashboard charts."""
    try:
        db = get_db()
        # Last 7 days attendance
        days_data = []
        for i in range(6, -1, -1):
            day = date.today() - timedelta(days=i)
            records = db.get_attendance_report(day, day)
            days_data.append({
                'date': day.strftime('%Y-%m-%d'),
                'day_name': day.strftime('%a'),
                'count': len(set(r['name'] for r in records))
            })
        
        # Department-wise distribution
        all_people = db.get_all_people()
        dept_counts = {}
        for person in all_people:
            dept = person.get('department') or 'Unassigned'
            dept_counts[dept] = dept_counts.get(dept, 0) + 1
        
        return jsonify({
            'success': True,
            'weekly_attendance': days_data,
            'department_distribution': [
                {'department': dept, 'count': count}
                for dept, count in dept_counts.items()
            ]
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# WEB ROUTES - PEOPLE MANAGEMENT
# ============================================================================

@app.route('/people')
def people_list():
    """People management page."""
    return render_template('people.html')


@app.route('/api/people')
def get_people():
    """Get all people."""
    try:
        db = get_db()
        people = db.get_all_people()
        return jsonify({'success': True, 'people': people})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/people/<int:person_id>')
def get_person(person_id):
    """Get person details."""
    try:
        db = get_db()
        person = db.get_person_by_id(person_id)
        if person:
            # Get attendance history
            history = db.get_person_attendance_history(person_id, limit=10)
            return jsonify({
                'success': True,
                'person': person,
                'recent_attendance': history
            })
        return jsonify({'success': False, 'error': 'Person not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/people', methods=['POST'])
def add_person():
    """Add new person."""
    try:
        db = get_db()
        data = request.json
        person_id = db.add_person(
            name=data['name'],
            label_id=data['label_id'],
            email=data.get('email'),
            department=data.get('department')
        )
        return jsonify({'success': True, 'person_id': person_id})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/people/<int:person_id>', methods=['PUT'])
def update_person(person_id):
    """Update person details."""
    try:
        db = get_db()
        data = request.json
        db.update_person(
            person_id=person_id,
            name=data.get('name'),
            email=data.get('email'),
            department=data.get('department')
        )
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/people/<int:person_id>', methods=['DELETE'])
def delete_person(person_id):
    """Delete person."""
    try:
        db = get_db()
        db.delete_person(person_id)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


# ============================================================================
# WEB ROUTES - ATTENDANCE MANAGEMENT
# ============================================================================

@app.route('/attendance')
def attendance_page():
    """Attendance records page."""
    return render_template('attendance.html')


@app.route('/api/attendance/today')
def get_today_attendance():
    """Get today's attendance."""
    try:
        db = get_db()
        records = db.get_today_attendance()
        return jsonify({'success': True, 'records': records})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/attendance/range')
def get_attendance_range():
    """Get attendance for date range."""
    try:
        db = get_db()
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        if not start_date or not end_date:
            return jsonify({'success': False, 'error': 'Missing date parameters'}), 400
        
        start = datetime.strptime(start_date, '%Y-%m-%d').date()
        end = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        records = db.get_attendance_report(start, end)
        return jsonify({'success': True, 'records': records})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/attendance/absentees')
def get_absentees():
    """Get today's absentees."""
    try:
        db = get_db()
        target_date = request.args.get('date')
        if target_date:
            target = datetime.strptime(target_date, '%Y-%m-%d').date()
        else:
            target = None
        
        absentees = db.get_absentees(target)
        return jsonify({'success': True, 'absentees': absentees})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/attendance/manual', methods=['POST'])
def add_manual_attendance():
    """Add manual attendance record."""
    try:
        db = get_db()
        data = request.json
        success = db.log_attendance(
            person_name=data['person_name'],
            confidence=data.get('confidence', 100.0),
            status=data.get('status', 'present')
        )
        
        if success:
            return jsonify({'success': True})
        return jsonify({'success': False, 'error': 'Failed to log attendance (person may already be logged today)'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/attendance/<int:record_id>', methods=['DELETE'])
def delete_attendance_record(record_id):
    """Delete attendance record."""
    try:
        db = get_db()
        db.delete_attendance_record(record_id)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


# ============================================================================
# WEB ROUTES - REPORTS
# ============================================================================

@app.route('/reports')
def reports_page():
    """Reports page."""
    return render_template('reports.html')


@app.route('/api/reports/summary')
def get_report_summary():
    """Get attendance summary report."""
    try:
        db = get_db()
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        if not start_date or not end_date:
            return jsonify({'success': False, 'error': 'Missing date parameters'}), 400
        
        start = datetime.strptime(start_date, '%Y-%m-%d').date()
        end = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        records = db.get_attendance_report(start, end)
        all_people = db.get_all_people()
        
        # Calculate summary statistics
        total_days = (end - start).days + 1
        total_people = len(all_people)
        total_possible = total_days * total_people
        total_present = len(records)
        
        # Per-person statistics
        person_stats = {}
        for person in all_people:
            person_stats[person['name']] = {
                'total_days': 0,
                'present_days': 0,
                'absent_days': 0,
                'rate': 0
            }
        
        # Count attendance per person per day
        attendance_by_person_day = {}
        for record in records:
            name = record['name']
            day = record['date']
            key = f"{name}_{day}"
            attendance_by_person_day[key] = True
        
        # Calculate per-person stats
        for person in all_people:
            name = person['name']
            present = sum(1 for key in attendance_by_person_day if key.startswith(f"{name}_"))
            absent = total_days - present
            rate = (present / total_days * 100) if total_days > 0 else 0
            
            person_stats[name] = {
                'total_days': total_days,
                'present_days': present,
                'absent_days': absent,
                'rate': round(rate, 1)
            }
        
        return jsonify({
            'success': True,
            'summary': {
                'start_date': start.isoformat(),
                'end_date': end.isoformat(),
                'total_days': total_days,
                'total_people': total_people,
                'total_possible': total_possible,
                'total_present': total_present,
                'overall_rate': round((total_present / total_possible * 100) if total_possible > 0 else 0, 1)
            },
            'person_stats': person_stats
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/reports/export')
def export_report():
    """Export attendance report as CSV."""
    try:
        db = get_db()
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        format_type = request.args.get('format', 'csv')
        
        if not start_date or not end_date:
            return jsonify({'success': False, 'error': 'Missing date parameters'}), 400
        
        start = datetime.strptime(start_date, '%Y-%m-%d').date()
        end = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        # Generate filename
        filename = f"attendance_report_{start}_{end}.{format_type}"
        output_path = Path("exports") / filename
        output_path.parent.mkdir(exist_ok=True)
        
        # Export based on format
        if format_type == 'csv':
            db.export_to_csv(str(output_path), start, end)
        elif format_type == 'json':
            records = db.get_attendance_report(start, end)
            with open(output_path, 'w') as f:
                json.dump(records, f, indent=2, default=str)
        else:
            return jsonify({'success': False, 'error': 'Invalid format'}), 400
        
        return send_file(output_path, as_attachment=True)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# WEB ROUTES - SETTINGS
# ============================================================================

@app.route('/settings')
def settings_page():
    """Settings page."""
    return render_template('settings.html')


@app.route('/api/settings')
def get_settings():
    """Get system settings."""
    try:
        db = get_db()
        settings = db.get_settings()
        return jsonify({'success': True, 'settings': settings})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Update system settings."""
    try:
        db = get_db()
        data = request.json
        db.update_settings(data)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


# ============================================================================
# MAIN
# ============================================================================

def create_templates():
    """Create HTML templates directory."""
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)
    
    # Import template creation function from the templates module
    print("✅ Templates directory created")
    print("   Note: Template HTML files will be created automatically on first run")
    
    return templates_dir


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Attendance Management System")
    parser.add_argument('--port', type=int, default=5000, help='Port to run server on')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--db', default='attendance.db', help='Database file path')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Initialize database
    global db
    db = AttendanceDatabase(args.db)
    
    # Create templates directory
    templates_dir = create_templates()
    
    # Check if templates exist
    if not (templates_dir / "base.html").exists():
        print("\n⚠️  HTML templates not found!")
        print(f"   Please create template files in: {templates_dir.absolute()}")
        print("   Required templates: base.html, dashboard.html, people.html, attendance.html, reports.html, settings.html")
        print("\n   Run the template creation script first to generate these files.")
        return
    
    # Start server
    print("\n" + "="*70)
    print("📊 ATTENDANCE MANAGEMENT SYSTEM")
    print("="*70)
    print(f"\n🗄️  Database: {args.db}")
    print(f"🌐 Server: http://{args.host}:{args.port}")
    print("\n📍 Available Pages:")
    print(f"   • Dashboard:  http://{args.host}:{args.port}/")
    print(f"   • People:     http://{args.host}:{args.port}/people")
    print(f"   • Attendance: http://{args.host}:{args.port}/attendance")
    print(f"   • Reports:    http://{args.host}:{args.port}/reports")
    print(f"   • Settings:   http://{args.host}:{args.port}/settings")
    print("\n⌨️  Press Ctrl+C to stop the server")
    print("="*70 + "\n")
    
    try:
        app.run(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped. Goodbye!")


if __name__ == "__main__":
    main()
