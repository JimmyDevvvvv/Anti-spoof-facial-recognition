# Test Report System - User Guide

**Automated Test Reporting for Anti-Spoofing System**  
**Version:** 1.0  
**Date:** October 22, 2025

---

## Overview

The test report system automatically captures and analyzes all test session data, generating comprehensive reports with visualizations and detailed metrics.

---

## Features

### 📊 **Automatic Data Collection**
- Records every frame processed
- Tracks all 10 anti-spoofing metrics
- Captures detection results (real vs spoof)
- Logs attack types detected
- Monitors performance metrics (FPS, processing time)
- Records recognition results (if enabled)
- Saves warnings and alerts

### 📈 **Visualizations**
- **Metrics Over Time**: Line graphs showing how each metric changes during the session
- **Detection Distribution**: Pie chart of real vs spoof detections
- **Attack Types**: Bar chart of attack types encountered
- **Metrics Heatmap**: Color-coded timeline of all metrics

### 📝 **Report Formats**
- **HTML Report**: Interactive web page with embedded visualizations
- **Text Report**: Human-readable summary for quick review
- **JSON Report**: Raw data for further analysis

---

## How to Use

### 🎬 **During Live Testing**

**Option 1: Generate Report Anytime**
```
Press 'G' key during testing
```
- Generates instant report of current session
- Testing continues after report is generated
- Can generate multiple reports during one session

**Option 2: Auto-Generate on Exit**
```
Press 'Q' or 'ESC' to quit
```
- Automatically generates final report
- Includes all data from entire session
- Saves before application closes

---

## Keyboard Controls (Updated)

| Key | Action |
|-----|--------|
| `G` | **Generate Report** (new!) |
| `1-4` | Change security level |
| `D` | Toggle debug info |
| `S` | Toggle statistics |
| `R` | Reset statistics |
| `SPACE` | Take screenshot |
| `Q/ESC` | Quit & generate final report |

---

## Report Contents

### 1. Session Information
```
- Start/End time
- Total duration
- Frames processed
- Security level used
```

### 2. Detection Results
```
- Real face count & percentage
- Spoof detection count & percentage
- Detection distribution chart
```

### 3. Attack Types Detected
```
- Printed Photo
- Digital Photo on Screen
- Video Replay
- 3D Mask
- Deepfake
- (etc.)

Shows count and percentage for each type
```

### 4. Metrics Summary (All 10 Metrics)
```
For each metric:
- Average score
- Minimum score
- Maximum score
- Standard deviation

Metrics include:
- Texture
- Motion
- Color
- Depth
- Frequency
- Blink
- Pulse
- Color Temperature (NEW)
- Screen Refresh (NEW)
- rPPG Blood Flow (NEW)
```

### 5. Performance Metrics
```
- Average FPS
- Average processing time
- Min/Max processing time
```

### 6. Warnings & Alerts
```
All warnings generated during session:
- "PHOTO DETECTED: ..."
- "SCREEN DETECTED: ..."
- "No heartbeat detected"
- (etc.)
```

### 7. Recognition Results (if enabled)
```
- Names recognized
- Recognition frequency
- Average confidence per person
```

### 8. Visualizations
```
- Metrics over time (4-panel chart)
- Detection distribution (pie + bar charts)
- Metrics heatmap (color timeline)
```

---

## Report Files Structure

```
test_reports/
├── session_20251022_143052/          # Auto-named by timestamp
│   ├── report.html                   # Open this in browser!
│   ├── report.txt                    # Quick text summary
│   ├── data.json                     # Raw data
│   └── visualizations/
│       ├── metrics_over_time.png
│       ├── detection_distribution.png
│       └── metrics_heatmap.png
```

---

## Usage Examples

### Example 1: Quick Test Session
```bash
# Start testing
python full-test.py --model .\models\combined_model.yml --level balanced

# Test with real face for 30 seconds
# Press 'G' to generate report
# Continue testing with printed photo
# Press 'G' again to compare results
# Press 'Q' to quit (generates final report)
```

**Result**: 3 reports generated
1. Real face test report
2. Printed photo test report  
3. Complete session report

---

### Example 2: Comprehensive Evaluation
```bash
# Start testing
python full-test.py --model .\models\combined_model.yml --level balanced

# Test sequence:
1. Real face (60 seconds)
2. Press 'G' → Baseline report
3. Printed photo (30 seconds)
4. Press 'G' → Photo detection report
5. Phone screen (30 seconds)
6. Press 'G' → Screen detection report
7. Press 'Q' → Final comprehensive report
```

**Result**: 4 detailed reports showing performance across different attack types

---

### Example 3: Performance Benchmarking
```bash
# Test all security levels
python full-test.py --model .\models\combined_model.yml --level lenient
# Test for 2 minutes, press 'G', then 'Q'

python full-test.py --model .\models\combined_model.yml --level balanced
# Test for 2 minutes, press 'G', then 'Q'

python full-test.py --model .\models\combined_model.yml --level strict
# Test for 2 minutes, press 'G', then 'Q'

python full-test.py --model .\models\combined_model.yml --level paranoid
# Test for 2 minutes, press 'G', then 'Q'
```

**Result**: 4 reports comparing:
- Detection rates across security levels
- Performance impact of each level
- False positive/negative rates

---

## Reading HTML Reports

### Opening the Report
1. Navigate to `test_reports/` folder
2. Open the session folder (e.g., `session_20251022_143052/`)
3. Double-click `report.html`
4. Opens in your default browser

### Report Sections
- **Header**: Session date/time
- **Stats Cards**: Key metrics at a glance
- **Charts**: Interactive visualizations
- **Tables**: Detailed breakdowns
- **Warnings**: Important alerts

### Colors
- 🟢 **Green**: Real face detections, good performance
- 🔴 **Red**: Spoof detections, warnings
- 🟡 **Yellow**: Moderate values, cautions
- 🔵 **Blue**: Headers, information

---

## Understanding Metrics in Reports

### High Scores = Good (Real Face)
- **Texture**: 0.6-0.9 → Natural skin texture
- **Color**: 0.6-0.9 → Natural skin tones
- **rPPG**: 0.7-1.0 → Clear heartbeat detected
- **Refresh**: 0.7-1.0 → No screen flicker

### Low Scores = Spoof Detected
- **Texture**: 0.0-0.3 → Smooth surface (paper/screen)
- **Color**: 0.0-0.3 → Flat/artificial colors
- **rPPG**: 0.0-0.3 → No heartbeat (photo/video)
- **Refresh**: 0.0-0.4 → Screen flicker detected

### Context-Dependent
- **Motion**: Can be high for both (hand movement)
- **Depth**: Varies with distance and pose
- **Blink**: Needs time to detect

---

## Troubleshooting

### Issue: No report generated on exit
**Solution**: Make sure session had at least 1 frame processed

### Issue: Visualizations missing
**Solution**: Install matplotlib:
```bash
pip install matplotlib
```

### Issue: Report generation slow
**Cause**: Many frames processed (>1000)  
**Normal**: Takes 5-10 seconds for visualization generation

### Issue: Can't open HTML report
**Solution**: Right-click → Open With → Browser (Chrome, Firefox, Edge)

---

## Advanced Usage

### Disable Reporting (if needed)
In `full-test.py`, line ~492:
```python
self.enable_reporting = False  # Set to False
```

### Change Report Output Directory
In `full-test.py`, line ~490:
```python
self.report_generator = TestReportGenerator(output_dir="my_reports")
```

### Export Data for Analysis
Use the `data.json` file:
```python
import json

with open('test_reports/session_20251022_143052/data.json', 'r') as f:
    data = json.load(f)

# Access metrics
for frame in data['metrics_history']:
    print(f"Frame {frame['frame']}: rPPG={frame['rppg_score']:.2f}")
```

---

## Report Metrics Explained

### Session Metrics
- **Duration**: Total test time in seconds
- **Total Frames**: Number of frames analyzed
- **Real/Spoof Rate**: Percentage of each detection type

### Performance Metrics
- **FPS**: Frames processed per second (target: 15-30)
- **Avg Processing Time**: Time per frame in ms (target: <50ms)

### Detection Accuracy
- **True Positive Rate**: Real faces correctly identified
- **False Positive Rate**: Spoofs incorrectly marked as real (target: <5%)
- **Attack Detection**: Specific attack types caught

---

## Best Practices

### 1. Generate Baseline Reports
- Test with real face for 60+ seconds
- Generate report with 'G'
- Save as reference for comparison

### 2. Test Each Attack Type Separately
- Test one attack type at a time
- Generate individual reports
- Compare metrics to identify patterns

### 3. Long-Duration Tests
- Run 5+ minute sessions for rPPG accuracy
- Blood flow detection needs time to stabilize
- More data = more accurate statistics

### 4. Document Test Conditions
- Note lighting conditions in filenames
- Save screenshots with SPACE key
- Keep reports organized by date/setup

### 5. Review Warnings Section
- Check for consistent warnings
- Indicates areas needing tuning
- Helps identify edge cases

---

## Report Examples

### Good Real Face Report
```
Detection Results:
  Real: 98.5%
  Spoof: 1.5%

Metrics Summary:
  Texture:    0.78 avg
  Color:      0.82 avg
  rPPG:       0.85 avg  ← Strong heartbeat
  Refresh:    0.92 avg

Warnings: None
```

### Printed Photo Detected
```
Detection Results:
  Real: 5.2%
  Spoof: 94.8%

Metrics Summary:
  Texture:    0.15 avg  ← Low (smooth paper)
  Color:      0.08 avg  ← Low (flat print)
  rPPG:       0.12 avg  ← No heartbeat!
  Refresh:    0.98 avg

Warnings:
  - PHOTO DETECTED: Low texture
  - PHOTO DETECTED: No blood flow detected
```

---

## Integration with Other Tools

### Export to Excel
Use `data.json` with pandas:
```python
import json
import pandas as pd

with open('data.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data['metrics_history'])
df.to_excel('report.xlsx', index=False)
```

### Compare Multiple Sessions
```python
import json
import glob

reports = glob.glob('test_reports/*/data.json')
for report in reports:
    with open(report, 'r') as f:
        data = json.load(f)
    print(f"{report}: {data['real_count']}/{data['total_frames']} real")
```

---

## Summary

The test report system provides:
- ✅ **Automatic** data collection during testing
- ✅ **Comprehensive** metrics and analysis
- ✅ **Visual** charts and graphs
- ✅ **Exportable** data in multiple formats
- ✅ **Historical** tracking of all sessions

**No manual logging needed!** Just test normally and press 'G' or quit to get your reports.

---

**Quick Start:**
1. Run: `python full-test.py --model .\models\combined_model.yml --level balanced`
2. Test your face/photos
3. Press `G` to generate report
4. Open `test_reports/[latest]/report.html` in browser
5. Done! 🎉

**Questions?** Check the generated reports for detailed analysis of your test sessions.
