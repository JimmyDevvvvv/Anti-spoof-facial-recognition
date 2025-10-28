# Blink Verification Guide (ADVISORY MODE)

## ⚠️ Important Notice

**Blink verification is now in ADVISORY MODE** - it monitors blink patterns and provides warnings but **DOES NOT reject recognition**. This is because:

1. Blink detection is experimental and not 100% reliable
2. The `UltimateAntiSpoof` system already handles blink-based liveness detection
3. False positives could block legitimate users
4. It's meant as an additional data point, not a hard requirement

## Overview

The Face Recognition system includes **Temporal Blink Verification** - an experimental feature that monitors blink patterns for additional security insights. It warns about suspicious patterns but delegates final liveness decisions to the main `UltimateAntiSpoof` system.

## How It Works

### Natural Blink Patterns (Reference)

Research shows that humans naturally blink:
- **Frequency**: 15-20 times per minute (every 3-4 seconds on average)
- **Min Interval**: At least 1-2 seconds between blinks
- **Variability**: Blink timing varies naturally

### Advisory Detection Method

The system tracks blink timestamps and provides feedback:

1. **Monitors blink frequency** 
   - Warns if impossibly fast (< 0.3s apart) - indicates video replay
   
2. **Tracks blink occurrence**
   - Notes if blinks are detected
   - Reports status for logging/debugging
   
3. **Does NOT reject recognition**
   - Warnings are logged but don't block legitimate users
   - Main anti-spoofing system makes final decision

## Configuration

### Default Settings (VERY LENIENT)

The system now uses very lenient defaults:
- **Min Blink Interval**: 0.5s (only catches impossibly fast blinks)
- **Max Blink Interval**: 20.0s (very forgiving)
- **Required Blinks**: 1 (just need one blink)
- **Verification Window**: 30.0s (long collection period)
- **Mode**: Advisory (warns but doesn't reject)

### Enable Blink Verification

```python
from src.face import FaceRecognizer

# Option 1: Enable during initialization (DISABLED BY DEFAULT)
recognizer = FaceRecognizer(
    enable_blink_verification=True,  # Must explicitly enable
    min_blink_interval=0.5,          # Very lenient
    max_blink_interval=20.0,         # Very lenient
    required_blinks=1,                # Just need 1 blink
    blink_window=30.0                # Long window
)

# Option 2: Enable after initialization
recognizer.enable_blink_verification_detection(
    min_blink_interval=0.5,
    max_blink_interval=20.0,
    required_blinks=1,
    blink_window=30.0
)
```

### Disable Blink Verification

```python
recognizer.disable_blink_verification_detection()
```

### Check Status

```python
info = recognizer.get_blink_verification_info()
print(f"Enabled: {info['enabled']}")
print(f"Current blinks: {info['current_blinks']}")
print(f"Verified: {info['blink_verified']}")
print(f"Session duration: {info['session_duration']:.1f}s")
```

## Integration with Anti-Spoofing

Blink verification works **alongside** the `UltimateAntiSpoof` system in advisory mode:

```python
recognizer = FaceRecognizer(
    enable_antispoofing=True,          # Main anti-spoofing (RECOMMENDED)
    antispoofing_mode="high_security",  # Use strict security
    enable_blink_verification=True      # Add advisory blink monitoring (OPTIONAL)
)

result = recognizer.predict_with_name(face_image)

# The main anti-spoofing system makes the final decision
if not result.is_live:
    print(f"Spoof detected: {result.name}")
    if result.liveness_result:
        print(f"Attack type: {result.liveness_result.attack_type.value}")

# Blink verification just provides additional info
blink_info = recognizer.get_blink_verification_info()
print(f"Blink status: {blink_info['current_blinks']} blinks detected")
```

**Important:** The `UltimateAntiSpoof` system has its own comprehensive blink detection using MediaPipe. The temporal blink verification is just an additional monitoring layer.

## Usage Examples

### Basic Recognition with Blink Verification

```python
import cv2
from src.face import FaceRecognizer

# Initialize recognizer
recognizer = FaceRecognizer(
    threshold=50.0,
    enable_antispoofing=True,
    enable_blink_verification=True
)

# Train model
recognizer.train(face_samples, labels, label_names)

# Recognition loop
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform recognition with blink verification
    result = recognizer.predict_with_name(frame)
    
    if result.recognized and result.is_live:
        print(f"✓ Recognized: {result.name}")
    elif not result.is_live:
        print(f"✗ Liveness check failed: {result.name}")
    else:
        print(f"✗ Unknown person")
    
    # Show blink verification status
    blink_info = recognizer.get_blink_verification_info()
    print(f"Blinks: {blink_info['current_blinks']}/{blink_info['required_blinks']}")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
```

### Custom Verification Parameters

```python
# Lenient settings (faster verification)
recognizer.enable_blink_verification_detection(
    min_blink_interval=1.0,   # Allow faster blinks
    max_blink_interval=15.0,  # Allow longer pauses
    required_blinks=1,         # Only need 1 blink
    blink_window=10.0         # Shorter time window
)

# Strict settings (slower but more secure)
recognizer.enable_blink_verification_detection(
    min_blink_interval=2.0,   # Require slower blinks
    max_blink_interval=8.0,   # Require more frequent blinks
    required_blinks=3,         # Need 3 blinks
    blink_window=20.0         # Longer time window
)
```

### Reset Between Sessions

```python
# Reset blink tracking for new person
recognizer.reset_blink_verification()

# Or reset everything (confidence + blinks)
recognizer.reset_confidence_history()  # This also resets blinks
```

## Recognition Result

When blink verification is enabled, it monitors but **does not reject**:

```python
result = recognizer.predict_with_name(face_image)

# Main recognition continues normally
if result.recognized and result.is_live:
    print(f"✓ Recognized: {result.name}")
    
# Blink warnings appear in console but don't block recognition
# Example console output:
#   ⚠️ Blink verification warning: Only 0/1 blinks in 30.0s (advisory only)
#   ✓ Recognized: John Doe  # Recognition still succeeds

# Check blink status programmatically
blink_info = recognizer.get_blink_verification_info()
print(f"Blinks detected: {blink_info['current_blinks']}")
print(f"Verified: {blink_info['blink_verified']}")
```

## Performance Considerations

### Verification Time

- **Initial delay**: 0-15 seconds (collecting blinks)
- **Processing overhead**: <5ms per frame
- **Memory usage**: Minimal (~100 bytes per blink)

### When to Use

✅ **Use blink verification (advisory mode) when:**
- You want additional logging/monitoring data
- Debugging anti-spoofing issues
- Gathering statistics on blink patterns
- Research or analysis purposes

❌ **Don't rely on it for:**
- Primary security (use `UltimateAntiSpoof` instead)
- Rejecting users (it's advisory only)
- Strict liveness detection (too many false positives)

## Recommended Configuration

**For most users:** Just use the main anti-spoofing system without blink verification:
```python
recognizer = FaceRecognizer(
    enable_antispoofing=True,
    antispoofing_mode="basic"  # or "high_security"
    # enable_blink_verification=False  # Default - blink monitoring disabled
)
```

**For advanced monitoring:**
```python
recognizer = FaceRecognizer(
    enable_antispoofing=True,
    antispoofing_mode="high_security",
    enable_blink_verification=True  # Enable for monitoring/logging only
)
```

## Best Practices

1. **Combine with UltimateAntiSpoof**: Use both systems for maximum security
2. **Adjust parameters**: Tune based on your specific use case
3. **Reset between users**: Call `reset_blink_verification()` for new sessions
4. **Monitor status**: Check `get_blink_verification_info()` for debugging
5. **User feedback**: Display blink count to guide users

## Troubleshooting

### "Insufficient blinks" Error

**Problem**: User didn't blink enough times
**Solution**: 
- Increase `blink_window` to give more time
- Decrease `required_blinks` to require fewer blinks
- Provide UI feedback asking user to blink

### "Blinks too frequent" Error

**Problem**: Detected as video replay or synthetic
**Solution**:
- Check if actual video replay attack
- Decrease `min_blink_interval` if false positive

### "Blinks too infrequent" Error

**Problem**: Detected as static image
**Solution**:
- Check if actual photo attack
- Increase `max_blink_interval` if false positive

## Technical Details

### Blink Detection Source

Blinks are detected by the `UltimateAntiSpoof` detector using:
- MediaPipe facial landmarks (78 points)
- Eye Aspect Ratio (EAR) algorithm
- Temporal analysis of eyelid movement

### State Management

The system maintains:
- `_blink_timestamps`: List of blink detection times
- `_session_start_time`: When verification session began
- `_blink_verified`: Whether verification passed
- `_verification_in_progress`: Active verification state

### Thread Safety

⚠️ **Warning**: Blink verification is NOT thread-safe. Use separate recognizer instances for concurrent processing.

## References

- Natural blink rates: Bentivoglio et al. (1997)
- Eye Aspect Ratio: Soukupová & Čech (2016)
- Liveness detection: ISO/IEC 30107-3
- MediaPipe Face Mesh: Google Research (2020)

## Support

For issues or questions:
1. Check `get_blink_verification_info()` status
2. Review this guide's troubleshooting section
3. Adjust parameters for your use case
4. Consider disabling if not needed
