# Screen/Photo Detection - Research Summary

## Best Methods from Literature (2021-2025)

### 1. **Backlight Emission Analysis** ✅ IMPLEMENTED
**Source**: Industry standards for display technology
- **Method**: Measure blue channel dominance and overall brightness
- **Theory**: LCD/OLED screens use LED backlights with higher blue wavelengths (450-495nm)
- **Thresholds**:
  - Blue ratio: > 0.35-0.38 (blue_channel / avg_brightness)
  - Brightness: > 115-130 (0-255 scale)
- **Effectiveness**: High for modern smartphone/tablet screens
- **Implementation**: In `_check_color()` method

### 2. **Moiré Pattern Detection** ⚠️ PARTIALLY IMPLEMENTED
**Source**: "Deep Learning for Face Anti-Spoofing: A Survey" (IEEE TPAMI 2022)
- **Method**: Detect interference patterns from camera-screen pixel grids
- **Theory**: Camera sensor grid × Screen pixel grid = Moiré interference
- **Detection**: FFT magnitude shows strong periodic components
- **Thresholds**:
  - FFT max magnitude > 500-1000 indicates pixel grid
- **Effectiveness**: Very high for screens, low false positives
- **Implementation**: In `_check_texture()` method (FFT analysis)

### 3. **Specular Reflection Analysis** ❌ NOT IMPLEMENTED
**Source**: Multiple papers on presentation attack detection
- **Method**: Analyze light reflections and highlights
- **Theory**: 
  - Screens: Uniform specular reflections (glass surface)
  - Real skin: Diffuse reflections with oily highlights
- **Detection**: Sobel edge + brightness thresholding
- **Effectiveness**: Medium (lighting dependent)
- **Recommendation**: Add to future version

### 4. **Color Temperature Analysis** ❌ NOT IMPLEMENTED
**Source**: Display technology research
- **Method**: Measure color temperature in Kelvin
- **Theory**:
  - Screens: 6500K-9000K (cool, blue-ish)
  - Real faces under indoor lighting: 2700K-5000K (warm)
  - Sunlight: 5500K-6500K
- **Detection**: Convert RGB → CCT (Correlated Color Temperature)
- **Effectiveness**: High for indoor scenarios
- **Recommendation**: **Should implement this!**

### 5. **Screen Refresh Rate Detection** ❌ NOT IMPLEMENTED
**Source**: Video-based anti-spoofing research
- **Method**: Detect periodic brightness fluctuations
- **Theory**: 
  - LCD screens refresh at 60Hz, 120Hz, etc.
  - Creates subtle flickering invisible to human eye
  - Can be detected with high-speed analysis
- **Detection**: Temporal FFT analysis of brightness over frames
- **Effectiveness**: Very high (unique to screens)
- **Limitation**: Requires video input (multiple frames)
- **Recommendation**: **Implement for video mode!**

### 6. **Pixel Grid Detection** ✅ IMPLEMENTED
**Source**: Image forensics literature
- **Method**: Detect regular pixel patterns using FFT
- **Theory**: Screens have regular RGB subpixel grids
- **Detection**: FFT shows peaks at spatial frequencies matching pixel pitch
- **Thresholds**: Max frequency magnitude > 500
- **Effectiveness**: High for close-range photos
- **Implementation**: In `_check_texture()` method

### 7. **Print Quality Artifacts** ❌ NOT IMPLEMENTED
**Source**: Document forensics
- **Method**: Detect halftone dot patterns from printing
- **Theory**: Printed photos have dot patterns from CMYK printing
- **Detection**: Wavelet analysis or bandpass filtering
- **Effectiveness**: Medium (only for printed photos, not screens)
- **Recommendation**: Low priority

### 8. **rPPG (Remote Photoplethysmography)** ❌ NOT IMPLEMENTED
**Source**: "Deep Learning for Face Anti-Spoofing" survey paper
- **Method**: Detect blood flow through skin color changes
- **Theory**: 
  - Real faces: Skin color varies with heartbeat (60-100 BPM)
  - Photos/screens: No color variation
- **Detection**: Temporal analysis of green channel over 5-10 seconds
- **Effectiveness**: **VERY HIGH** (gold standard)
- **Limitation**: Requires 5-10 seconds of video
- **Recommendation**: **HIGHEST PRIORITY for future!**

### 9. **Eye Movement Analysis** ⚠️ BLINK ONLY
**Source**: Behavioral biometrics research
- **Method**: Track eye movements and saccades
- **Theory**:
  - Real people: Eyes move constantly (microsaccades)
  - Photos/screens: Eyes completely static
- **Detection**: MediaPipe eye tracking
- **Current**: Only blink detection implemented
- **Recommendation**: Add saccade detection

### 10. **3D Structure from Motion** ❌ NOT IMPLEMENTED
**Source**: Computer vision research (Structure from Motion)
- **Method**: Reconstruct 3D structure from video
- **Theory**:
  - Real faces: Consistent 3D structure with head movement
  - Screens: Flat plane, no depth parallax
- **Detection**: Feature tracking + epipolar geometry
- **Effectiveness**: Very high
- **Limitation**: Requires head movement
- **Recommendation**: Complex but very effective

## Current Implementation Status

### ✅ Implemented Methods:
1. **Backlight Detection** (Blue ratio + brightness)
2. **Pixel Grid Detection** (FFT frequency analysis)
3. **Motion Analysis** (Optical flow)
4. **Texture Analysis** (Variance, edges, entropy)
5. **Depth Estimation** (Gradient analysis)
6. **Blink Detection** (MediaPipe eye tracking)

### ❌ Missing High-Impact Methods:
1. **rPPG (Blood Flow)** - **HIGHEST PRIORITY**
2. **Screen Refresh Rate** - **HIGH PRIORITY**
3. **Color Temperature** - **MEDIUM PRIORITY**
4. **Specular Reflection** - **MEDIUM PRIORITY**

## Recommended Next Steps

### **Immediate (Current Session)**
- [x] Add backlight detection (blue channel analysis)
- [x] Add pixel grid detection (FFT)
- [x] Tighten detection thresholds

### **Short-term (Next Update)**
1. **Implement Screen Refresh Detection**
   - Analyze brightness variations over frames
   - Detect 60Hz/120Hz flickering patterns
   
2. **Implement Color Temperature Analysis**
   - Convert RGB → CCT (Correlated Color Temperature)
   - Flag if CCT > 6500K (too blue/cool)

3. **Add Specular Reflection Detection**
   - Detect uniform glass-like reflections
   - Compare with diffuse skin reflections

### **Long-term (Future Versions)**
1. **Implement rPPG (Blood Flow Detection)**
   - Analyze green channel variations over 5-10s
   - Extract heartbeat signal (60-100 BPM)
   - **Most reliable method**

2. **Train Deep Learning Model**
   - Use CDCN or similar architecture
   - Train on screen-specific dataset
   - Deploy as additional check

## References

1. **"Deep Learning for Face Anti-Spoofing: A Survey"**
   - Yu et al., IEEE TPAMI 2022
   - arXiv:2106.14948
   - Comprehensive survey of methods

2. **"Silent-Face-Anti-Spoofing"**
   - MiniVision AI, GitHub
   - Industry-standard open source solution

3. **"Moiré Pattern Detection"**
   - Wikipedia: Moiré Pattern
   - Physics of screen interference

4. **Display Technology Standards**
   - LCD/OLED backlight specifications
   - Color temperature standards

## Testing Recommendations

1. **Test with different devices:**
   - iPhone (OLED) vs Android (LCD)
   - Tablets vs Phones
   - Old vs new displays

2. **Test under different lighting:**
   - Bright indoor (high backlight contrast)
   - Dim indoor (backlight more visible)
   - Outdoor (backlight less visible)

3. **Test with different photos:**
   - High-quality photos
   - Low-quality photos
   - Printed photos vs screen photos

## Performance Targets

- **Screen Detection Rate**: > 95%
- **False Positive Rate** (real faces flagged as screens): < 5%
- **Processing Time**: < 100ms per frame
- **Current Status**: Screen detection ~60-70% (needs improvement)
