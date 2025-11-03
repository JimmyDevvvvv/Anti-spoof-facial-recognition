# 📊 Per-User Reporting System - Quick Summary

## What's New?

The face recognition system now **automatically generates detailed per-user reports** every time you run it! 

## ✅ Key Features Added

1. **Automatic Report Generation**
   - Reports generated automatically when system exits
   - No manual configuration needed - works out of the box!

2. **Per-User Statistics**
   - Individual metrics for OMAR, NOUR, and all other users
   - Recognition rates, confidence scores, quality metrics
   - Anti-spoofing detection per user

3. **Real-time Tracking**
   - Every recognition attempt logged immediately
   - Processing times, quality scores, confidence values
   - Spoof detection tracking

4. **Multiple Export Formats**
   - **JSON** - For programmatic access and data analysis
   - **Text** - Human-readable reports for easy review

5. **Persistent Storage**
   - SQLite database stores all session history
   - Query past sessions anytime
   - No data loss between runs

6. **Auto-Save Feature**
   - Periodic auto-saves every 50 attempts
   - Reports saved even if system crashes
   - Manual report generation with 'G' key

## 📁 Where Are Reports Saved?

```
Face/
├── reports/                          # All reports here!
│   ├── user_report_20251029_011455.json
│   ├── user_report_20251029_011455.txt
│   └── user_report_20251029_011455_auto.json
└── user_reports.db                   # SQLite database
```

## 🚀 How to Use

### 1. Run the System Normally

```bash
python full-test.py --model models/combined_model.yml
```

That's it! Reports are **automatically enabled**.

### 2. Exit to Generate Report

Press **Q** or **ESC** to exit → Report automatically generated!

### 3. View Your Report

```bash
# PowerShell
Get-Content reports\user_report_20251029_011455.txt

# Or just open the .txt file in any text editor
```

## 📊 What's in the Report?

### For Each User (OMAR, NOUR, etc.):

✅ Total recognition attempts  
✅ Successful recognitions  
✅ Recognition rate (%)  
✅ Average confidence scores  
✅ Confidence range (min/max)  
✅ Average quality scores  
✅ Average processing time  
✅ **Anti-spoofing statistics:**
   - Total checks
   - Real vs spoof detections
   - Spoof types (photo/video/mask)
✅ **Edge case performance:**
   - Low quality detections
   - Low/high confidence cases
✅ Sample confidence values (first 10)

### Overall Session Summary:

✅ Session ID and duration  
✅ Total attempts across all users  
✅ Overall accuracy  
✅ Total spoofs detected  
✅ Number of unknown attempts  

## 📖 Example Report

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

  Sample Confidences (First 10): 29.5, 32.1, 30.8, 28.9, 33.2, ...
```

## 🎮 Controls

While system is running:
- **Q** or **ESC** - Exit and generate final report
- **G** - Generate report immediately (system keeps running)
- **S** - Screenshot
- **D** - Toggle debug info

## 🔍 Advanced Usage

### View JSON Report Programmatically

```python
import json

with open('reports/user_report_20251029_011455.json', 'r') as f:
    report = json.load(f)

# Access user data
omar_stats = report['per_user_details']['OMAR']
print(f"OMAR Recognition Rate: {omar_stats['recognition_rate']}%")
```

### Convert to CSV/Excel

```python
import json
import pandas as pd

with open('reports/user_report_20251029_011455.json', 'r') as f:
    report = json.load(f)

df = pd.DataFrame.from_dict(report['per_user_details'], orient='index')
df.to_csv('user_statistics.csv')
df.to_excel('user_statistics.xlsx')
```

## 📚 Full Documentation

For complete documentation including:
- Configuration options
- Database schema
- Integration guides
- Troubleshooting
- API reference

**See:** [docs/PER_USER_REPORTING.md](PER_USER_REPORTING.md)

## 💡 Why This is Useful

✅ **Academic Documentation** - Perfect for research papers and presentations  
✅ **Performance Monitoring** - Track system accuracy over time  
✅ **User Analysis** - Understand per-user recognition patterns  
✅ **Security Auditing** - Monitor spoof detection effectiveness  
✅ **Quality Assurance** - Identify problematic conditions or users  
✅ **Data for Improvement** - Use statistics to improve model training  

## 🎯 Quick Test

Want to see it in action? Run the test script:

```bash
python src\face\per_user_report_generator.py
```

This generates a sample report with test data in the `reports/` directory!

---

**✨ That's it! Reports are now automatically generated every time you run the system. No extra configuration needed!**

For questions or issues, check the [full documentation](PER_USER_REPORTING.md).
