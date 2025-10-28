# Quick Test Guide - Screen Detection

## 🧪 **How to Test**

### **Test 1: Phone Screen (Should be SPOOF)**
1. Open a photo of a face on your phone
2. Run the system:
   ```bash
   python full-test.py --model .\models\combined_model.yml --level balanced
   ```
3. Hold phone in front of webcam
4. **Expected Results**:
   - Status: `[ERROR] SPOOF`
   - ColorTemp: 0.2-0.4 (cool blue backlight)
   - Refresh: 0.2-0.5 (60Hz flicker detected)
   - Color: 0.1-0.3 (blue excess)
   - Warning: `SCREEN DETECTED: Cool color temp; Screen refresh detected`

### **Test 2: Real Face (Should be REAL)**
1. Look at webcam normally
2. Wait 5 seconds for rPPG to initialize
3. **Expected Results**:
   - Status: `[OK] REAL`
   - ColorTemp: 0.7-1.0 (warm natural lighting)
   - Refresh: 0.8-1.0 (no flicker)
   - Color: 0.6-1.0 (normal skin)
   - rPPG: 0.8-1.0 after 5s (heartbeat detected)

### **Test 3: Printed Photo (Should be SPOOF)**
1. Print a photo and hold it up
2. **Expected Results**:
   - Status: `[ERROR] SPOOF`
   - Texture: 0.3-0.5 (flat, less variation)
   - Depth: 0.2-0.4 (no 3D structure)
   - Motion: 0.0-0.2 (completely static)
   - Pulse: 0.3 (no micro-expressions)

---

## 🎛️ **Keyboard Controls**

- **L**: Switch to LENIENT mode
- **B**: Switch to BALANCED mode
- **S**: Switch to STRICT mode
- **D**: Toggle debug panel
- **R**: Toggle recognition
- **Q/ESC**: Quit

---

## 📊 **What to Look For**

### **Screen Indicators**:
1. **Color Temperature < 0.4**: Blue LED backlight detected
2. **Refresh Score < 0.4**: 60Hz/120Hz flicker detected
3. **Color Score < 0.3**: Excessive blue channel

**If 2+ indicators**: Automatic SCREEN OVERRIDE → SPOOF

### **Debug Output**:
```
[DEBUG] Color Temperature: 6850K
[DEBUG] SCREEN TEMP DETECTED: 6850K (typical LCD)
[DEBUG] SCREEN REFRESH DETECTED: 60Hz flicker (peak=450.2, baseline=85.3)
[DEBUG] ⚠ SCREEN OVERRIDE: ['Cool color temp (screen backlight)', 'Screen refresh detected']
```

---

## 🐛 **Troubleshooting**

### **Problem**: Real face detected as spoof
**Solutions**:
- Ensure good lighting (not too bright/blue)
- Move slightly to generate motion
- Wait 5 seconds for rPPG to stabilize
- Switch to LENIENT mode (press 'L')

### **Problem**: Screen not detected
**Solutions**:
- Ensure video mode is enabled (default: ON)
- Wait 30 frames for refresh detection
- Check screen brightness (higher = better detection)
- Enable debug mode (press 'D') to see scores

### **Problem**: rPPG always shows 0.5
**Reason**: Need 5 seconds (150 frames) of data
**Solution**: Wait patiently, it will update

---

## 📈 **Performance Expectations**

| Metric | Target | Actual |
|--------|--------|--------|
| Screen Detection Rate | >95% | Test now! |
| False Positive Rate | <5% | Test now! |
| Real-time FPS | >25 | ~30 fps |
| Processing Time | <35ms | ~25ms |

---

## 🎯 **Success Criteria**

✅ **System is working correctly if**:
1. Phone screens → Detected as SPOOF
2. Real faces → Detected as REAL
3. Printed photos → Detected as SPOOF
4. Debug shows correct temperature/refresh values
5. No crashes or errors

❌ **System needs tuning if**:
1. Real faces consistently marked as SPOOF
2. Phone screens marked as REAL
3. Scores all show 0.5 (not updating)
4. Crashes or errors occur
