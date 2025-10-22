# Anti-Spoofing Face Recognition System

This document describes the integrated anti-spoofing capabilities of the face recognition system, providing comprehensive protection against presentation attacks.

## 🛡️ Overview

The anti-spoofing module provides multi-layered liveness detection to prevent various types of presentation attacks:

- **Print Attacks**: Photos printed on paper
- **Digital Display Attacks**: Photos/videos shown on screens
- **Video Replay Attacks**: Pre-recorded videos
- **Mask Attacks**: 3D masks or cutouts
- **Photo Cutout Attacks**: Cut-out photos held up to camera

## 🔧 Integration with Face Recognizer

The anti-spoofing functionality is seamlessly integrated into the `FaceRecognizer` class:

```python
from src.face import FaceRecognizer

# Create recognizer with anti-spoofing enabled
recognizer = FaceRecognizer(
    threshold=50.0,
    enable_antispoofing=True,
    antispoofing_mode="basic",  # or "high_security"
    reject_on_spoof=True
)

# Recognition now includes liveness checks
result = recognizer.predict_with_name(face_image)

if result.recognized and result.is_live:
    print(f"✓ Live person recognized: {result.name}")
elif not result.is_live:
    print(f"✗ Spoofing attack detected: {result.liveness_result.detected_attack_type}")
else:
    print("✗ Person not recognized")
```

## 🎯 Detection Methods

### 1. Texture Analysis (LBP)
- **Purpose**: Detect smooth surfaces typical of prints/screens
- **Method**: Local Binary Pattern analysis
- **Effectiveness**: High against print attacks

### 2. Frequency Domain Analysis
- **Purpose**: Detect moiré patterns and print artifacts
- **Method**: FFT analysis for periodic patterns
- **Effectiveness**: High against digital displays and prints

### 3. Color Space Analysis
- **Purpose**: Verify natural skin tone properties
- **Method**: YCrCb color space skin detection
- **Effectiveness**: Good against unnatural color reproduction

### 4. Depth Estimation
- **Purpose**: Detect 3D structure vs flat surfaces
- **Method**: Edge gradient and shading analysis
- **Effectiveness**: High against photo attacks

### 5. Motion Analysis (Video Mode)
- **Purpose**: Detect natural micro-movements
- **Method**: Optical flow analysis
- **Effectiveness**: High against static photos

### 6. Eye Blink Detection (Video Mode)
- **Purpose**: Verify natural blinking patterns
- **Method**: Eye Aspect Ratio (EAR) tracking
- **Effectiveness**: Very high against static attacks

## ⚙️ Configuration Options

### Anti-Spoofing Modes

#### Basic Mode
```python
recognizer = FaceRecognizer(
    enable_antispoofing=True,
    antispoofing_mode="basic"
)
```
- **Features**: Texture, frequency, color, depth analysis
- **Speed**: Fast (single image)
- **Security**: Good for general use
- **Use Case**: Standard security applications

#### High Security Mode
```python
recognizer = FaceRecognizer(
    enable_antispoofing=True,
    antispoofing_mode="high_security"
)
```
- **Features**: All basic features + stricter thresholds
- **Speed**: Fast (single image)
- **Security**: High security
- **Use Case**: Critical security applications

### Standalone Detectors

For applications that need only anti-spoofing without recognition:

```python
from src.face import create_basic_detector, create_high_security_detector

# Basic detector
detector = create_basic_detector(strict_mode=False)
result = detector.detect(face_image)

# High security detector
detector = create_high_security_detector(require_blink=False, require_motion=False)
result = detector.detect(face_image)
```

## 📊 Detection Results

The anti-spoofing system returns comprehensive results:

```python
result = recognizer.predict_with_name(face_image)

# Basic information
print(f"Recognized: {result.recognized}")
print(f"Is Live: {result.is_live}")
print(f"Confidence: {result.confidence:.2%}")

# Detailed liveness information
if result.liveness_result:
    liveness = result.liveness_result
    print(f"Liveness Level: {liveness.liveness_level.value}")
    print(f"Attack Type: {liveness.detected_attack_type.value if liveness.detected_attack_type else 'None'}")
    print(f"Processing Time: {liveness.processing_time:.3f}s")
    
    # Individual metric results
    if liveness.texture_metrics:
        print(f"Texture Score: {liveness.texture_metrics.lbp_score:.2f}")
    if liveness.frequency_metrics:
        print(f"Moiré Score: {liveness.frequency_metrics.moiré_score:.2f}")
    if liveness.color_metrics:
        print(f"Skin Tone Score: {liveness.color_metrics.skin_tone_score:.2f}")
```

## 🎮 Usage Examples

### Example 1: Basic Integration
```python
from src.face import FaceRecognizer

# Load trained model
recognizer = FaceRecognizer(enable_antispoofing=True)
recognizer.load_model("models/your_model.yml")

# Process camera feed
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    faces = detect_faces(frame)  # Your face detection
    
    for face in faces:
        result = recognizer.predict_with_name(face)
        
        if result.recognized and result.is_live:
            print(f"✓ {result.name} (Live)")
        elif not result.is_live:
            print(f"✗ Spoofing detected: {result.liveness_result.detected_attack_type}")
```

### Example 2: Dynamic Configuration
```python
# Start without anti-spoofing
recognizer = FaceRecognizer(enable_antispoofing=False)

# Enable basic anti-spoofing
recognizer.enable_antispoofing_detection(mode="basic", reject_on_spoof=True)

# Switch to high security
recognizer.enable_antispoofing_detection(mode="high_security", reject_on_spoof=True)

# Disable anti-spoofing
recognizer.disable_antispoofing_detection()

# Get current configuration
info = recognizer.get_antispoofing_info()
print(f"Anti-spoofing enabled: {info['enabled']}")
print(f"Mode: {info['mode']}")
```

### Example 3: Video Stream Analysis
```python
from src.face import create_video_detector

# Create video-based detector
detector = create_video_detector(require_blink=True, strict_mode=True)

# Analyze video stream
result = detector.detect_video_stream(
    video_source=0,  # Camera
    duration_seconds=3.0,
    require_blink=True,
    require_motion=True
)

print(f"Live: {result.is_live}")
print(f"Blinks detected: {result.blink_metrics.blink_count}")
print(f"Motion detected: {result.motion_metrics.face_motion_detected}")
```

## 🔍 Advanced Features

### Challenge-Response Anti-Spoofing
```python
from src.face.antispoofing import ChallengeResponseAntiSpoofing

detector = create_basic_detector()
challenge_system = ChallengeResponseAntiSpoofing(detector)

# Perform random challenge
success, message = challenge_system.perform_challenge(camera_id=0)
print(f"Challenge result: {message}")

# Multi-challenge verification
all_passed = challenge_system.multi_challenge_verification(
    video_source=0,
    num_challenges=2
)
```

### Adaptive Threshold Adjustment
```python
from src.face.antispoofing import AdaptiveAntiSpoofing

detector = create_basic_detector()
adaptive_detector = AdaptiveAntiSpoofing(detector)

# Detect with feedback for learning
result = adaptive_detector.detect_with_feedback(
    face_image,
    ground_truth=True  # True if actually live, False if spoof
)

# Get performance statistics
stats = adaptive_detector.get_performance_stats()
print(f"False positive rate: {stats['fp_rate']:.2%}")
print(f"False negative rate: {stats['fn_rate']:.2%}")
```

## 📈 Performance Considerations

### Speed vs Security Trade-offs

| Mode | Speed | Security | Use Case |
|------|-------|----------|----------|
| Disabled | Fastest | None | Development/testing |
| Basic | Fast | Good | General applications |
| High Security | Fast | High | Critical applications |
| Video Mode | Slower | Very High | High-security applications |

### Optimization Tips

1. **Single Image Mode**: Use for real-time applications
2. **Video Mode**: Use for high-security applications where delay is acceptable
3. **Strict Mode**: Enable for maximum security, disable for better user experience
4. **Threshold Tuning**: Adjust confidence thresholds based on your false positive/negative requirements

## 🚨 Security Considerations

### Limitations
- **Sophisticated Attacks**: Advanced 3D masks or high-quality video replays may still bypass detection
- **Lighting Conditions**: Very poor lighting may affect detection accuracy
- **Camera Quality**: Low-resolution cameras may reduce effectiveness

### Best Practices
1. **Multi-layered Security**: Combine with other security measures
2. **Regular Updates**: Keep detection algorithms updated
3. **User Education**: Train users to recognize spoofing attempts
4. **Monitoring**: Log and monitor detection results for analysis

## 🧪 Testing and Validation

### Test Different Attack Types
```python
# Test with various attack types
test_images = {
    "live_face.jpg": True,
    "printed_photo.jpg": False,
    "screen_display.jpg": False,
    "video_replay.jpg": False
}

for image_path, expected_live in test_images.items():
    result = detector.detect(cv2.imread(image_path))
    print(f"{image_path}: Expected {expected_live}, Got {result.is_live}")
```

### Batch Testing
```python
from src.face.antispoofing import batch_test_images

# Test folder of images
results = batch_test_images(
    image_folder="test_images/",
    output_csv="test_results.csv"
)
```

## 📚 API Reference

### FaceRecognizer Anti-Spoofing Methods

- `enable_antispoofing_detection(mode, reject_on_spoof)`: Enable anti-spoofing
- `disable_antispoofing_detection()`: Disable anti-spoofing
- `get_antispoofing_info()`: Get current configuration

### Anti-Spoofing Detector Methods

- `detect(face_image)`: Single image detection
- `detect_video_stream(video_source, duration)`: Video stream analysis
- `get_detector_info()`: Get detector configuration
- `reset_state()`: Reset internal state

### Factory Functions

- `create_basic_detector(strict_mode)`: Basic detector
- `create_high_security_detector()`: High-security detector
- `create_video_detector()`: Video-based detector
- `create_lightweight_detector()`: Lightweight detector

## 🚀 Quick Start

### Installation
```bash
# Required dependencies (already have these)
pip install opencv-python numpy

# Optional for better performance
pip install scikit-image  # For advanced LBP
pip install scipy  # For signal processing
```

### 30-Second Integration
```python
from src.face import create_basic_detector

# Create detector
detector = create_basic_detector(strict_mode=True)

# Test an image
result = detector.detect(face_image)

if result.is_live:
    print("✓ Live person - proceed with recognition")
    # Your recognition code here
else:
    print(f"✗ Spoofing detected: {result.detected_attack_type}")
    # Reject the authentication attempt
```

## 🔗 Integration Methods

### Method 1: Preprocessing Integration (Recommended)
Integrate directly into your preprocessing pipeline:
```python
from src.face.preprocessing import create_production_pipeline
from src.face.antispoofing import create_basic_detector, integrate_with_preprocessor

# Create preprocessor
preprocessor = create_production_pipeline(target_size=(112, 112))

# Create detector
detector = create_basic_detector(strict_mode=True)

# Integrate them
preprocessor = integrate_with_preprocessor(preprocessor, detector)

# Now preprocessing includes anti-spoofing
result = preprocessor.preprocess_with_antispoofing(
    face_image,
    face_bbox=(x, y, w, h)
)

if result.spoofing_metrics.is_live:
    # Use result.processed_image for recognition
    processed = result.processed_image
else:
    print(f"Rejected: {result.spoofing_metrics.detected_attack_type}")
```

### Method 2: Recognizer Wrapper (Easiest)
Wrap your existing recognizer:
```python
from src.face.antispoofing import create_antispoofing_recognizer, create_high_security_detector

# Your existing recognizer
class MyFaceRecognizer:
    def recognize(self, face_image):
        # Your recognition logic
        return {'identity': 'John', 'confidence': 0.95}

# Create secure version
detector = create_high_security_detector()
secure_recognizer = create_antispoofing_recognizer(
    MyFaceRecognizer(),
    detector,
    reject_on_spoof=True
)

# Use it (automatic spoofing check)
result = secure_recognizer.recognize(face_image)

if result['success']:
    print(f"Recognized: {result['identity']}")
    print(f"Liveness: {result['liveness'].confidence:.1%}")
else:
    print(f"Rejected: {result['reason']}")
```

### Method 3: Manual Integration
Full control over the process:
```python
from src.face.antispoofing import create_video_detector

detector = create_video_detector(require_blink=True, strict_mode=True)

def secure_face_recognition(face_image, recognizer):
    """Your custom integration."""
    
    # Step 1: Anti-spoofing check
    liveness_result = detector.detect(face_image)
    
    if not liveness_result.is_live:
        return {
            'success': False,
            'reason': f'Spoofing detected: {liveness_result.detected_attack_type}',
            'liveness_confidence': liveness_result.confidence
        }
    
    # Step 2: Preprocessing
    preprocessed = your_preprocessing(face_image)
    
    # Step 3: Recognition
    recognition_result = recognizer.predict(preprocessed)
    
    # Step 4: Adjust confidence based on liveness
    adjusted_confidence = (
        recognition_result['confidence'] * 
        liveness_result.confidence
    )
    
    return {
        'success': True,
        'identity': recognition_result['identity'],
        'confidence': adjusted_confidence,
        'liveness_confidence': liveness_result.confidence,
        'liveness_level': liveness_result.liveness_level.value
    }
```

## 💡 Usage Examples

### Example 1: Single Image Authentication
```python
from src.face.antispoofing import create_basic_detector
import cv2

# Setup
detector = create_basic_detector(strict_mode=True)

# Load image
face_image = cv2.imread('user_photo.jpg')

# Detect spoofing
result = detector.detect(face_image)

# Detailed analysis
print(result.get_detailed_report())

# Check result
if result.is_live:
    if result.confidence >= 0.85:
        print("High confidence - proceed")
    elif result.confidence >= 0.65:
        print("Medium confidence - request additional verification")
    else:
        print("Low confidence - reject")
else:
    print(f"REJECTED - Attack type: {result.detected_attack_type}")
    
    # Log the attempt
    log_spoofing_attempt(
        attack_type=result.detected_attack_type,
        confidence=result.confidence,
        timestamp=datetime.now()
    )
```

### Example 2: Video Stream Authentication
```python
from src.face.antispoofing import create_video_detector

# For high-security scenarios
detector = create_video_detector(require_blink=True, strict_mode=True)

# Analyze video stream
result = detector.detect_video_stream(
    video_source=0,  # Webcam
    duration_seconds=3.0,
    require_blink=True,
    require_motion=True
)

print(f"Status: {result.liveness_level.value}")
print(f"Confidence: {result.confidence:.1%}")

if result.blink_metrics:
    print(f"Blinks detected: {result.blink_metrics.blink_count}")
    print(f"Blink frequency: {result.blink_metrics.blink_frequency:.2f}/s")

if result.motion_metrics:
    print(f"Motion detected: {result.motion_metrics.face_motion_detected}")
    print(f"Motion magnitude: {result.motion_metrics.optical_flow_magnitude:.2f}")
```

### Example 3: Challenge-Response Verification
```python
from src.face.antispoofing import ChallengeResponseAntiSpoofing, create_video_detector

# Create challenge system
detector = create_video_detector()
challenge_system = ChallengeResponseAntiSpoofing(detector)

# Perform multi-challenge verification
success = challenge_system.multi_challenge_verification(
    video_source=0,
    num_challenges=2
)

if success:
    print("✓ User verified - all challenges passed")
    # Proceed with recognition
else:
    print("✗ Verification failed")
    # Reject authentication
```

### Example 4: Batch Processing
```python
from src.face.antispoofing import create_basic_detector
import cv2
from pathlib import Path

detector = create_basic_detector(strict_mode=True)

# Process multiple images
image_folder = Path("user_submissions")
results = []

for img_path in image_folder.glob("*.jpg"):
    image = cv2.imread(str(img_path))
    result = detector.detect(image)
    
    results.append({
        'filename': img_path.name,
        'is_live': result.is_live,
        'confidence': result.confidence,
        'attack_type': result.detected_attack_type
    })
    
    # Take action
    if result.is_live:
        # Move to verified folder
        verified_path = Path("verified") / img_path.name
        img_path.rename(verified_path)
    else:
        # Move to rejected folder
        rejected_path = Path("rejected") / img_path.name
        img_path.rename(rejected_path)

# Summary
live_count = sum(1 for r in results if r['is_live'])
print(f"Verified: {live_count}/{len(results)}")
```

### Example 5: Adaptive Threshold Learning
```python
from src.face.antispoofing import AdaptiveAntiSpoofing, create_basic_detector

# Create adaptive system
base_detector = create_basic_detector()
adaptive_detector = AdaptiveAntiSpoofing(base_detector)

# Process with feedback
face_image = cv2.imread('test_image.jpg')
ground_truth = True  # You know this is actually a live person

result = adaptive_detector.detect_with_feedback(
    face_image,
    ground_truth=ground_truth
)

# System automatically adjusts thresholds based on feedback
stats = adaptive_detector.get_performance_stats()
print(f"Current threshold: {stats['current_threshold']:.2f}")
print(f"False positive rate: {stats['fp_rate']:.2%}")
print(f"False negative rate: {stats['fn_rate']:.2%}")
```

## ⚙️ Performance Tuning

### Detection Modes Comparison

| Mode | Speed | Security | Use Case |
|------|-------|----------|----------|
| PASSIVE_SINGLE | ⚡⚡⚡ | ⚡⚡ | Real-time applications |
| PASSIVE_MULTI | ⚡⚡ | ⚡⚡⚡ | High-security applications |
| ACTIVE_BLINK | ⚡ | ⚡⚡⚡⚡ | Critical security |
| ACTIVE_MOTION | ⚡ | ⚡⚡⚡⚡ | Maximum security |
| ACTIVE_CHALLENGE | ⚡ | ⚡⚡⚡⚡⚡ | Ultra-high security |

### Threshold Optimization

```python
# Adjust confidence thresholds based on your requirements
detector = create_basic_detector(strict_mode=False)

# For high-security applications
detector.confidence_threshold = 0.75

# For general applications
detector.confidence_threshold = 0.60

# For lenient applications
detector.confidence_threshold = 0.45
```

### Performance Optimization Tips

1. **Single Image Mode**: Use for real-time applications
2. **Video Mode**: Use for high-security applications where delay is acceptable
3. **Strict Mode**: Enable for maximum security, disable for better user experience
4. **Threshold Tuning**: Adjust confidence thresholds based on your false positive/negative requirements

## 🎯 Conclusion

The integrated anti-spoofing system provides robust protection against common presentation attacks while maintaining ease of use and flexibility. Choose the appropriate security level based on your application requirements and performance constraints.

For more examples and advanced usage, see `examples/antispoofing_recognition_demo.py`.
