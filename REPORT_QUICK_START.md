# Test Report System - Quick Reference

## 🎯 What You Get

**Automatic test reports with:**
- 📊 All 10 anti-spoofing metrics tracked
- 📈 Beautiful visualizations (charts, graphs, heatmaps)
- 📝 Multiple formats (HTML, text, JSON)
- ⚡ Performance analysis (FPS, processing time)
- 🎭 Attack type detection breakdown
- 👤 Recognition results (if enabled)

---

## 🚀 How to Use

### During Testing:

```
Press 'G' → Generate report instantly
Press 'Q' → Quit and auto-generate final report
```

### Find Your Reports:

```
test_reports/
└── session_20251022_143052/
    ├── report.html      ← Open this in browser!
    ├── report.txt       ← Quick summary
    ├── data.json        ← Raw data
    └── visualizations/
        ├── metrics_over_time.png
        ├── detection_distribution.png
        └── metrics_heatmap.png
```

---

## 📊 Report Contents

### Session Info
- Duration, frames processed
- Real vs spoof detection rates

### All 10 Metrics Analyzed
- Texture, Motion, Color, Depth, Frequency
- Blink, Pulse
- **Color Temperature** (NEW)
- **Screen Refresh** (NEW)
- **rPPG Blood Flow** (NEW)

### Performance
- FPS, processing time
- Min/max/average stats

### Attack Types
- Which attacks detected
- Frequency of each type

### Visualizations
- Metrics timeline
- Detection pie chart
- Heatmap of all metrics

---

## 🎨 Reading Reports

**Open** `report.html` in any browser

**Colors:**
- 🟢 Green = Real face / Good
- 🔴 Red = Spoof / Warning
- 🟡 Yellow = Moderate / Caution

**Key Metrics:**
- **rPPG > 0.7** = Real (heartbeat detected)
- **rPPG < 0.3** = Spoof (no heartbeat)
- **Texture > 0.6** = Real (natural skin)
- **Texture < 0.3** = Spoof (smooth surface)

---

## 💡 Use Cases

### 1. Compare Real vs Spoof
```
1. Test with real face → Press 'G'
2. Test with photo → Press 'G'
3. Compare the two reports!
```

### 2. Benchmark Security Levels
```
Test lenient → Press 'G'
Test balanced → Press 'G'
Test strict → Press 'G'
Compare detection rates!
```

### 3. Track Performance
```
Run 5-minute test → Press 'Q'
Check FPS and processing time in report
```

---

## 🔧 Updated Controls

| Key | Action |
|-----|--------|
| **G** | **Generate Report** ⭐ NEW |
| 1-4 | Security levels |
| D | Toggle debug |
| S | Toggle stats |
| R | Reset stats |
| Q/ESC | Quit (auto-report) |

---

## 📁 Files Added

1. **`generate_report.py`**
   - Report generation engine
   - Visualization creator

2. **`TEST_REPORT_GUIDE.md`**
   - Complete user guide
   - Advanced usage

3. **`full-test.py`** (updated)
   - Integrated report system
   - Press 'G' to generate

---

## ✅ No Extra Steps Needed!

**It just works:**
- Testing automatically recorded
- Reports generated on demand
- All data saved for analysis

**Try it now:**
```bash
python full-test.py --model .\models\combined_model.yml --level balanced
# Test for a minute
# Press 'G'
# Open the HTML report!
```

---

## 🎓 Example Report Output

```
======================================================================
REPORT GENERATED: session_20251022_143052
======================================================================
Location: test_reports\session_20251022_143052
Files:
  - json: test_reports\session_20251022_143052\data.json
  - text: test_reports\session_20251022_143052\report.txt
  - html: test_reports\session_20251022_143052\report.html
  - visualizations: test_reports\session_20251022_143052\visualizations
======================================================================
```

**Open `report.html` and see:**
- Beautiful web interface
- All metrics visualized
- Complete session analysis
- Professional-looking report

---

## 🎉 That's It!

No configuration needed. Just press 'G' during testing or quit normally to get comprehensive reports of your anti-spoofing tests!

**See `TEST_REPORT_GUIDE.md` for detailed documentation.**
