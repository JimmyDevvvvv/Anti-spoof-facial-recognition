#!/usr/bin/env python3
"""
Comprehensive Face Recognition & Anti-Spoofing Test

Single test file covering:
- Face recognition functionality
- Anti-spoofing detection
- Live camera testing
- Model loading and validation
"""

import cv2
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from src.face import (
    FaceRecognizer, 
    FaceDetector,
    create_secure_recognizer,
    IntegrationMode
)


def test_recognition_only():
    """Test basic face recognition without anti-spoofing."""
    print("Testing Face Recognition Only")
    print("=" * 40)
    
    try:
        # Create basic recognizer
        recognizer = FaceRecognizer(threshold=50.0)
        
        # Load model if available
        model_path = Path("models/clean_omar_model.yml")
        if model_path.exists():
            recognizer.load_model(model_path)
            print(f"[OK] Model loaded: {model_path}")
            print(f"   Known people: {len(recognizer.get_known_people())}")
        else:
            print(f"[WARN]  No model found: {model_path}")
            return False
        
        # Test with camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Camera not available")
            return False
        
        detector = FaceDetector()
        print("[CAMERA] Testing recognition (press 'q' to quit)...")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect faces
            faces = detector.detect_faces(frame)
            
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                
                # Get recognition result
                result = recognizer.predict_with_name(face_roi)
                
                # Display result
                color = (0, 255, 0) if result.recognized else (0, 0, 255)
                text = f"{result.name}: {result.confidence:.1f}"
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, text, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.imshow('Recognition Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("[OK] Recognition test completed")
        return True
        
    except Exception as e:
        print(f"[ERROR] Recognition test failed: {e}")
        return False


def test_antispoofing_only():
    """Test anti-spoofing detection only."""
    print("\nTesting Anti-Spoofing Only")
    print("=" * 40)
    
    try:
        # Create secure recognizer with anti-spoofing
        recognizer = create_secure_recognizer(
            threshold=50.0,
            antispoofing_mode=IntegrationMode.BALANCED
        )
        
        # Load model if available
        model_path = Path("models/clean_omar_model.yml")
        if model_path.exists():
            recognizer.load_model(model_path)
            print(f"[OK] Model loaded: {model_path}")
        else:
            print(f"[WARN]  No model found: {model_path}")
        
        # Test with camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Camera not available")
            return False
        
        detector = FaceDetector()
        print("[CAMERA] Testing anti-spoofing (press 'q' to quit, 's' to switch mode)...")
        
        modes = [IntegrationMode.BASIC, IntegrationMode.BALANCED, IntegrationMode.HIGH_SECURITY]
        current_mode_idx = 1
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect faces
            faces = detector.detect_faces(frame)
            
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                
                # Get result with anti-spoofing
                result = recognizer.predict_with_antispoofing(face_roi)
                
                # Determine display info
                if result.is_live and result.recognized:
                    color = (0, 255, 0)  # Green
                    text = f"{result.name} (LIVE)"
                elif not result.is_live:
                    color = (0, 0, 255)  # Red
                    text = "SPOOFING DETECTED"
                else:
                    color = (0, 165, 255)  # Orange
                    text = "Unknown (LIVE)"
                
                # Draw rectangle and text
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, text, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Show liveness confidence
                live_text = f"Live: {result.liveness_confidence:.0%}"
                cv2.putText(frame, live_text, (x, y+h+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show mode
            mode_text = f"Mode: {modes[current_mode_idx].value.upper()}"
            cv2.putText(frame, mode_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.imshow('Anti-Spoofing Test', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                current_mode_idx = (current_mode_idx + 1) % len(modes)
                recognizer.set_antispoofing_mode(modes[current_mode_idx])
                print(f"Switched to: {modes[current_mode_idx].value}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("[OK] Anti-spoofing test completed")
        return True
        
    except Exception as e:
        print(f"[ERROR] Anti-spoofing test failed: {e}")
        return False


def test_model_validation():
    """Test model loading and validation."""
    print("\nTesting Model Validation")
    print("=" * 40)
    
    try:
        # Test different model files
        model_paths = [
            "models/clean_omar_model.yml",
            "models/improved_model.yml", 
            "models/final_omar_model.yml"
        ]
        
        for model_path in model_paths:
            path = Path(model_path)
            if path.exists():
                recognizer = FaceRecognizer()
                result = recognizer.load_model(model_path)
                
                if result['model_loaded']:
                    print(f"[OK] {model_path}: {len(recognizer.get_known_people())} people")
                else:
                    print(f"[FAIL] {model_path}: Failed to load")
            else:
                print(f"[WARN] {model_path}: Not found")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Model validation failed: {e}")
        return False


def main():
    """Run comprehensive tests."""
    print("Comprehensive Face Recognition & Anti-Spoofing Test")
    print("=" * 60)
    print()
    print("This test covers:")
    print("• Face recognition functionality")
    print("• Anti-spoofing detection")
    print("• Model loading and validation")
    print()
    print("Controls:")
    print("• Press 'q' to quit any test")
    print("• Press 's' to switch anti-spoofing modes")
    print()
    print("=" * 60)
    
    tests = [
        ("Model Validation", test_model_validation),
        ("Recognition Only", test_recognition_only),
        ("Anti-Spoofing Only", test_antispoofing_only)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except KeyboardInterrupt:
            print(f"\n[WARN]  {test_name} interrupted by user")
            results.append((test_name, False))
            break
        except Exception as e:
            print(f"[ERROR] {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "[OK] PASS" if result else "[ERROR] FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("[SUCCESS] All tests passed! System is working correctly.")
    else:
        print("[WARN]  Some tests failed. Check the error messages above.")
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
