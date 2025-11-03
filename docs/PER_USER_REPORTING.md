# Per-User Reporting System

## Overview

The Face Recognition System now includes automatic **per-user reporting** that generates detailed reports for every recognition session. This feature tracks individual user performance, anti-spoofing statistics, and system metrics in real-time.

## Features

✅ **Automatic Report Generation** - Reports are generated automatically when the system exits  
✅ **Per-User Statistics** - Detailed metrics for each recognized user  
✅ **Real-time Tracking** - Every recognition attempt is logged in real-time  
✅ **Anti-Spoofing Metrics** - Track spoof detection rates per user  
✅ **Persistent Storage** - SQLite database stores all session data  
✅ **Multiple Export Formats** - JSON and human-readable text reports  
✅ **Auto-Save Feature** - Periodic auto-saves every 50 attempts  

## How It Works

### Automatic Integration

The reporting system is **automatically enabled** when you run the face recognition system:

```bash
python full-test.py --model models/combined_model.yml
```

### What Gets Tracked

For **each user**, the system tracks:

- ✅ Total recognition attempts
- ✅ Successful recognitions
- ✅ Failed attempts
- ✅ Recognition rate (%)
- ✅ Average confidence scores
- ✅ Confidence range (min/max)
- ✅ Average quality scores
- ✅ Average processing time
- ✅ Anti-spoofing checks (real vs spoof detections)
- ✅ Spoof types detected (photo/video/mask)
- ✅ Edge case performance (low quality, confidence issues)
- ✅ Sample confidence values

### Report Generation

Reports are generated in **two ways**:

1. **Automatic** - When you exit the system (press Q or ESC)
2. **Manual** - Press 'G' during runtime to generate a report immediately

## Report Output

### Directory Structure

```
Face/
├── reports/                           # All reports saved here
│   ├── user_report_20251029_011455.json
│   ├── user_report_20251029_011455.txt
│   └── user_report_20251029_011455_auto.json  # Auto-saved reports
├── user_reports.db                    # SQLite database (persistent storage)
└── full-test.py
```

### Text Report Format

```
================================================================================
DETAILED PER-USER RECOGNITION REPORT
================================================================================

SESSION INFORMATION
--------------------------------------------------------------------------------
Session ID: 20251029_011455
Start Time: 2025-10-29T01:14:55
Duration: 156.3s
Total Attempts: 125
Unknown Attempts: 8

OVERALL SUMMARY
--------------------------------------------------------------------------------
Total Users Recognized: 3
Overall Accuracy: 93.6%
Total Spoofs Detected: 5

================================================================================
PER-USER DETAILED RESULTS
================================================================================

User: OMAR
--------------------------------------------------------------------------------
  Total Attempts: 58
  Successful Recognitions: 55
  Failed Attempts: 3
  Recognition Rate: 94.83%

  Average Confidence: 31.25
  Confidence Range: 27.8 - 37.2
  Average Quality Score: 0.82
  Average Processing Time: 98.5ms

  Anti-Spoofing Checks:
    Total Checks: 58
    Real Detections: 56
    Spoof Detections: 2
    Detection Rate: 3.45%
    Spoof Types: photo(1), video(1)

  Edge Case Performance:
    Low Quality Detections: 4
    Low Confidence Cases: 2
    High Confidence Cases: 48

  Sample Confidences (First 10): 29.5, 32.1, 30.8, ...

User: NOUR
--------------------------------------------------------------------------------
  Total Attempts: 45
  Successful Recognitions: 44
  Failed Attempts: 1
  Recognition Rate: 97.78%
  ...
```

### JSON Report Format

The JSON report contains the same data in structured format for programmatic access:

```json
{
  "session_info": {
    "session_id": "20251029_011455",
    "start_time": "2025-10-29T01:14:55",
    "duration_seconds": 156.3,
    "total_attempts": 125,
    "unknown_attempts": 8
  },
  "per_user_details": {
    "OMAR": {
      "total_attempts": 58,
      "successful_recognitions": 55,
      "recognition_rate": 94.83,
      "average_confidence": 31.25,
      "anti_spoofing": { ... },
      ...
    },
    "NOUR": { ... }
  },
  "summary": {
    "total_users": 3,
    "overall_accuracy": 93.6,
    "total_spoofs_detected": 5
  }
}
```

## Configuration

### Auto-Save Interval

You can configure how often auto-saves occur by modifying `full-test.py`:

```python
self.report_generator = PerUserReportGenerator(
    db_path="user_reports.db",
    reports_dir="reports",
    auto_save_interval=50  # Change this number (default: 50 attempts)
)
```

### Report Location

Change where reports are saved:

```python
self.report_generator = PerUserReportGenerator(
    reports_dir="my_custom_reports_folder"  # Custom directory
)
```

## Database Schema

The system uses SQLite for persistent storage with three tables:

### Sessions Table
- `session_id` (TEXT) - Unique session identifier
- `start_time` (REAL) - Session start timestamp
- `end_time` (REAL) - Session end timestamp
- `total_attempts` (INTEGER) - Total recognition attempts
- `unknown_attempts` (INTEGER) - Attempts with no recognition

### Recognition Attempts Table
- `id` (INTEGER) - Auto-incrementing primary key
- `session_id` (TEXT) - Foreign key to sessions
- `timestamp` (REAL) - Attempt timestamp
- `user_name` (TEXT) - Recognized user name
- `recognized` (BOOLEAN) - Whether recognition succeeded
- `confidence` (REAL) - Recognition confidence score
- `quality_score` (REAL) - Image quality score
- `processing_time` (REAL) - Processing time in seconds
- `is_spoof` (BOOLEAN) - Whether spoof was detected
- `spoof_type` (TEXT) - Type of spoof (photo/video/mask)

### User Statistics Table
- `session_id` (TEXT) - Foreign key to sessions
- `user_name` (TEXT) - User name
- `total_attempts` (INTEGER) - Total attempts for this user
- `successful_recognitions` (INTEGER) - Successful recognitions
- `avg_confidence` (REAL) - Average confidence score
- `avg_quality` (REAL) - Average quality score
- `avg_processing_time` (REAL) - Average processing time
- `anti_spoofing_checks` (INTEGER) - Total anti-spoofing checks
- `spoofs_detected` (INTEGER) - Number of spoofs detected

## Usage Examples

### Basic Usage

```bash
# Run with automatic reporting (enabled by default)
python full-test.py --model models/combined_model.yml

# Reports will be automatically generated when you exit (Q or ESC)
```

### Manual Report Generation

While the system is running:
- Press **G** to generate a report immediately
- The system continues running after generating the report
- Reports are saved with the current timestamp

### Viewing Reports

```bash
# View text report
cat reports/user_report_20251029_011455.txt

# View JSON report
cat reports/user_report_20251029_011455.json

# PowerShell
Get-Content reports\user_report_20251029_011455.txt
```

### Programmatic Access

```python
import json

# Load JSON report
with open('reports/user_report_20251029_011455.json', 'r') as f:
    report = json.load(f)

# Access per-user data
omar_stats = report['per_user_details']['OMAR']
print(f"OMAR Recognition Rate: {omar_stats['recognition_rate']}%")
print(f"OMAR Average Confidence: {omar_stats['average_confidence']}")
```

## Integration with Other Systems

### Using the Report Generator Standalone

You can use the report generator in your own scripts:

```python
from src.face.per_user_report_generator import PerUserReportGenerator

# Initialize
generator = PerUserReportGenerator()

# Log recognition attempts
generator.log_recognition_attempt(
    user_name="OMAR",
    recognized=True,
    confidence=28.5,
    quality_score=0.85,
    processing_time=0.095
)

# Generate report
report = generator.finalize_session()
```

### Export to Other Formats

The JSON format can be easily converted to CSV, Excel, or other formats:

```python
import json
import pandas as pd

# Load JSON report
with open('reports/user_report_20251029_011455.json', 'r') as f:
    report = json.load(f)

# Convert to DataFrame
df = pd.DataFrame.from_dict(report['per_user_details'], orient='index')

# Export to CSV
df.to_csv('user_statistics.csv')

# Export to Excel
df.to_excel('user_statistics.xlsx')
```

## Performance Considerations

- **Minimal Overhead**: Logging adds < 1ms per recognition attempt
- **Efficient Storage**: SQLite database with indexed queries
- **Auto-Save**: Periodic saves prevent data loss
- **Async-Ready**: Report generation doesn't block recognition

## Troubleshooting

### Reports Not Being Generated

**Problem**: No reports in the `reports/` directory

**Solutions**:
1. Check if reporting is enabled:
   ```python
   print(f"Reporting enabled: {system.enable_reporting}")
   ```
2. Ensure you properly exit (Q or ESC) - Ctrl+C may not trigger cleanup
3. Check for write permissions in the `reports/` directory

### Database Locked Errors

**Problem**: `sqlite3.OperationalError: database is locked`

**Solutions**:
1. Close any other programs accessing `user_reports.db`
2. Delete the database file to start fresh: `del user_reports.db`
3. Check for multiple instances of the program running

### Missing User Data

**Problem**: Some users not appearing in reports

**Solutions**:
1. Check if recognition was successful (confidence < threshold)
2. Verify anti-spoofing didn't reject the face
3. Check console output for recognition results

## Best Practices

1. ✅ **Regular Backups**: Backup `user_reports.db` and `reports/` directory periodically
2. ✅ **Review Reports**: Check reports after each session for system performance
3. ✅ **Monitor Accuracy**: Track recognition rates to detect model degradation
4. ✅ **Spoof Analysis**: Review spoof detection patterns to improve security
5. ✅ **Archive Old Reports**: Move old reports to archive directory to keep `reports/` clean

## Future Enhancements

Planned features for future versions:

- 📊 **Web Dashboard**: Real-time visualization of statistics
- 📧 **Email Reports**: Automatic email delivery of session reports
- 📈 **Trend Analysis**: Compare sessions over time
- 🔔 **Alerts**: Notifications for suspicious activity or low accuracy
- 🌐 **API Integration**: RESTful API for remote report access
- 📱 **Mobile App**: View reports on mobile devices

## Support

For questions or issues with the reporting system:
1. Check the console output for error messages
2. Review the generated reports for data accuracy
3. Examine `user_reports.db` with a SQLite viewer
4. Check file permissions in the `reports/` directory

---

**Note**: This reporting system is designed for development, testing, and production monitoring. It provides comprehensive insights into system performance and user-specific behavior.
