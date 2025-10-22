#!/usr/bin/env python3
"""
OpenCV Version Compatibility Checker
====================================

This script checks all face recognition modules for OpenCV version compatibility
and ensures they work with opencv-contrib-python==4.10.0.84
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Fix Unicode encoding on Windows
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def check_opencv_version():
    """Check OpenCV version and cv2.face module availability."""
    print("="*60)
    print("OPENCV VERSION COMPATIBILITY CHECK")
    print("="*60)
    
    # Check OpenCV version
    opencv_version = cv2.__version__
    print(f"OpenCV Version: {opencv_version}")
    
    # Check if it's the correct version
    if opencv_version.startswith("4.10"):
        print("✅ OpenCV version is correct (4.10.x)")
    else:
        print(f"❌ OpenCV version mismatch! Expected 4.10.x, got {opencv_version}")
        print("   Run: pip uninstall opencv-python opencv-contrib-python -y")
        print("   Then: pip install opencv-contrib-python==4.10.0.84")
        return False
    
    # Check cv2.face module
    if hasattr(cv2, 'face'):
        print("✅ cv2.face module is available")
        
        # Check LBPHFaceRecognizer
        if hasattr(cv2.face, 'LBPHFaceRecognizer_create'):
            print("✅ LBPHFaceRecognizer_create is available")
        else:
            print("❌ LBPHFaceRecognizer_create not found")
            return False
    else:
        print("❌ cv2.face module not available")
        print("   This means opencv-contrib-python is not installed correctly")
        return False
    
    return True

def test_face_recognition_modules():
    """Test all face recognition modules for compatibility."""
    print("\n" + "="*60)
    print("TESTING FACE RECOGNITION MODULES")
    print("="*60)
    
    modules_to_test = [
        ("detector", "src.face.detector", "FaceDetector"),
        ("preprocessing", "src.face.preprocessing", "EnhancedFacePreprocessor"),
        ("recognizer", "src.face.recognizer", "FaceRecognizer"),
        ("antispoofing", "src.face.antispoofing", "AntiSpoofingDetector"),
        ("antispoofing_integration", "src.face.antispoofing_integration", "PerfectAntiSpoof")
    ]
    
    results = {}
    
    for module_name, import_path, class_name in modules_to_test:
        print(f"\nTesting {module_name}...")
        try:
            # Import the module
            module = __import__(import_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            
            # Try to create an instance
            if module_name == "recognizer":
                instance = cls(threshold=50.0)
            elif module_name == "antispoofing_integration":
                instance = cls(level="balanced")
            else:
                instance = cls()
            
            print(f"✅ {module_name} - Import and instantiation successful")
            results[module_name] = True
            
        except Exception as e:
            print(f"❌ {module_name} - Error: {e}")
            results[module_name] = False
    
    return results

def test_opencv_functions():
    """Test critical OpenCV functions used in the modules."""
    print("\n" + "="*60)
    print("TESTING OPENCV FUNCTIONS")
    print("="*60)
    
    # Create a test image
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    test_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    
    functions_to_test = [
        ("cv2.CascadeClassifier", lambda: cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')),
        ("cv2.cvtColor", lambda: cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)),
        ("cv2.resize", lambda: cv2.resize(test_image, (50, 50))),
        ("cv2.Laplacian", lambda: cv2.Laplacian(test_gray, cv2.CV_64F)),
        ("cv2.Sobel", lambda: cv2.Sobel(test_gray, cv2.CV_64F, 1, 0, ksize=3)),
        ("cv2.Canny", lambda: cv2.Canny(test_gray, 50, 150)),
        ("cv2.calcOpticalFlowFarneback", lambda: cv2.calcOpticalFlowFarneback(test_gray, test_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)),
        ("cv2.face.LBPHFaceRecognizer_create", lambda: cv2.face.LBPHFaceRecognizer_create()),
    ]
    
    results = {}
    
    for func_name, test_func in functions_to_test:
        try:
            result = test_func()
            print(f"✅ {func_name} - Working")
            results[func_name] = True
        except Exception as e:
            print(f"❌ {func_name} - Error: {e}")
            results[func_name] = False
    
    return results

def test_mediapipe_compatibility():
    """Test MediaPipe compatibility."""
    print("\n" + "="*60)
    print("TESTING MEDIAPIPE COMPATIBILITY")
    print("="*60)
    
    try:
        import mediapipe as mp
        print("✅ MediaPipe is available")
        
        # Test MediaPipe Face Mesh
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("✅ MediaPipe Face Mesh can be initialized")
        return True
        
    except ImportError:
        print("⚠️ MediaPipe not available - this is optional")
        print("   Install with: pip install mediapipe>=0.10.0")
        return False
    except Exception as e:
        print(f"❌ MediaPipe error: {e}")
        return False

def generate_compatibility_report():
    """Generate a comprehensive compatibility report."""
    print("\n" + "="*60)
    print("COMPATIBILITY REPORT")
    print("="*60)
    
    # Check OpenCV
    opencv_ok = check_opencv_version()
    
    # Test modules
    module_results = test_face_recognition_modules()
    
    # Test OpenCV functions
    function_results = test_opencv_functions()
    
    # Test MediaPipe
    mediapipe_ok = test_mediapipe_compatibility()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_modules_ok = all(module_results.values())
    all_functions_ok = all(function_results.values())
    
    if opencv_ok and all_modules_ok and all_functions_ok:
        print("🎉 ALL TESTS PASSED!")
        print("✅ OpenCV version is correct")
        print("✅ All face recognition modules are compatible")
        print("✅ All OpenCV functions are working")
        if mediapipe_ok:
            print("✅ MediaPipe is available (optional)")
        else:
            print("⚠️ MediaPipe not available (optional)")
        print("\n🚀 Your face recognition system is ready to use!")
        return True
    else:
        print("❌ SOME TESTS FAILED!")
        if not opencv_ok:
            print("❌ OpenCV version issue")
        if not all_modules_ok:
            print("❌ Some modules have issues")
        if not all_functions_ok:
            print("❌ Some OpenCV functions have issues")
        
        print("\n🔧 FIXES NEEDED:")
        if not opencv_ok:
            print("1. Fix OpenCV version:")
            print("   pip uninstall opencv-python opencv-contrib-python -y")
            print("   pip install opencv-contrib-python==4.10.0.84")
        
        return False

if __name__ == "__main__":
    success = generate_compatibility_report()
    sys.exit(0 if success else 1)
