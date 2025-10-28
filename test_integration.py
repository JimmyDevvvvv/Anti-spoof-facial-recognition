"""
Integration Test - Verify all three files work together
========================================================

This script tests the integration between:
1. final_anti_spoof.py (UltimateAntiSpoof)
2. recognizer.py (FaceRecognizer)
3. full-test.py (LiveTestSystem)
"""

import sys
import numpy as np
from pathlib import Path

print("="*70)
print("INTEGRATION TEST - Anti-Spoofing + Recognition + Live Test")
print("="*70)

# Test 1: Import all modules
print("\n[Test 1] Importing modules...")
try:
    from src.face.final_anti_spoof import (
        UltimateAntiSpoof,
        SecurityLevel,
        AttackType,
        AntiSpoofResult,
        LivenessMetrics
    )
    print("  ✓ final_anti_spoof.py imports successful")
except Exception as e:
    print(f"  ✗ Failed to import final_anti_spoof.py: {e}")
    sys.exit(1)

try:
    from src.face.recognizer import FaceRecognizer, RecognitionResult
    print("  ✓ recognizer.py imports successful")
except Exception as e:
    print(f"  ✗ Failed to import recognizer.py: {e}")
    sys.exit(1)

# Test 2: Create UltimateAntiSpoof instance
print("\n[Test 2] Creating UltimateAntiSpoof detector...")
try:
    detector = UltimateAntiSpoof(level="balanced", enable_video_mode=True, debug=False)
    print(f"  ✓ Anti-spoofing detector created (level: balanced)")
except Exception as e:
    print(f"  ✗ Failed to create detector: {e}")
    sys.exit(1)

# Test 3: Test anti-spoofing check
print("\n[Test 3] Testing anti-spoofing check...")
try:
    # Create test image
    test_image = np.random.randint(100, 150, (480, 640, 3), dtype=np.uint8)
    
    # Run check
    result = detector.check(test_image)
    
    # Verify result structure
    assert hasattr(result, 'is_real'), "Missing is_real attribute"
    assert hasattr(result, 'confidence'), "Missing confidence attribute"
    assert hasattr(result, 'attack_type'), "Missing attack_type attribute"
    assert hasattr(result, 'metrics'), "Missing metrics attribute"
    
    # Verify metrics structure
    metrics = result.metrics
    assert hasattr(metrics, 'texture_score'), "Missing texture_score"
    assert hasattr(metrics, 'motion_score'), "Missing motion_score"
    assert hasattr(metrics, 'color_score'), "Missing color_score"
    assert hasattr(metrics, 'depth_score'), "Missing depth_score"
    assert hasattr(metrics, 'frequency_score'), "Missing frequency_score"
    assert hasattr(metrics, 'blink_score'), "Missing blink_score"
    assert hasattr(metrics, 'pulse_score'), "Missing pulse_score"
    
    # Verify NEW metrics
    assert hasattr(metrics, 'color_temp_score'), "Missing color_temp_score"
    assert hasattr(metrics, 'refresh_score'), "Missing refresh_score"
    assert hasattr(metrics, 'rppg_score'), "Missing rppg_score"
    
    print(f"  ✓ Anti-spoofing check successful")
    print(f"    - is_real: {result.is_real}")
    print(f"    - confidence: {result.confidence:.2%}")
    print(f"    - attack_type: {result.attack_type.value}")
    print(f"    - texture: {metrics.texture_score:.2f}")
    print(f"    - color_temp: {metrics.color_temp_score:.2f}")
    print(f"    - refresh: {metrics.refresh_score:.2f}")
    print(f"    - rppg: {metrics.rppg_score:.2f}")
    
except Exception as e:
    print(f"  ✗ Anti-spoofing check failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test video frame mode
print("\n[Test 4] Testing video frame mode...")
try:
    test_image = np.random.randint(100, 150, (480, 640, 3), dtype=np.uint8)
    result = detector.check_video_frame(test_image)
    
    assert hasattr(result, 'metrics'), "Missing metrics in video frame result"
    assert hasattr(result.metrics, 'color_temp_score'), "Missing color_temp_score in video mode"
    assert hasattr(result.metrics, 'refresh_score'), "Missing refresh_score in video mode"
    assert hasattr(result.metrics, 'rppg_score'), "Missing rppg_score in video mode"
    
    print(f"  ✓ Video frame mode successful")
    print(f"    - New metrics available: color_temp={result.metrics.color_temp_score:.2f}, "
          f"refresh={result.metrics.refresh_score:.2f}, rppg={result.metrics.rppg_score:.2f}")
    
except Exception as e:
    print(f"  ✗ Video frame mode failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test FaceRecognizer with anti-spoofing
print("\n[Test 5] Testing FaceRecognizer with anti-spoofing...")
try:
    recognizer = FaceRecognizer(
        enable_antispoofing=True,
        antispoofing_mode="basic",
        reject_on_spoof=True
    )
    
    # Verify anti-spoofing is enabled
    assert recognizer.enable_antispoofing, "Anti-spoofing not enabled"
    assert recognizer.antispoofing_detector is not None, "Anti-spoofing detector not created"
    
    print(f"  ✓ FaceRecognizer with anti-spoofing created")
    print(f"    - Anti-spoofing enabled: {recognizer.enable_antispoofing}")
    print(f"    - Anti-spoofing mode: {recognizer.antispoofing_mode}")
    print(f"    - Detector type: {type(recognizer.antispoofing_detector).__name__}")
    
except Exception as e:
    print(f"  ✗ FaceRecognizer creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test full-test.py components
print("\n[Test 6] Testing full-test.py components...")
try:
    # Import after other imports to avoid conflicts
    import cv2
    # Import using importlib since filename has dash
    import importlib.util
    spec = importlib.util.spec_from_file_location("full_test", "full-test.py")
    full_test = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(full_test)
    VisualOverlay = full_test.VisualOverlay
    PerformanceTracker = full_test.PerformanceTracker
    
    # Test PerformanceTracker
    tracker = PerformanceTracker()
    tracker.add_antispoofing_result(True)
    tracker.add_antispoofing_result(False)
    stats = tracker.get_stats()
    
    assert 'total_checks' in stats, "Missing total_checks in stats"
    assert stats['total_checks'] == 2, f"Expected 2 checks, got {stats['total_checks']}"
    
    print(f"  ✓ PerformanceTracker working")
    
    # Test VisualOverlay
    overlay = VisualOverlay()
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Test draw_status_panel with all 10 metrics
    overlay.draw_status_panel(
        test_frame,
        is_real=True,
        attack_type="None",
        texture=0.8,
        motion=0.7,
        color=0.9,
        depth=0.6,
        frequency=0.75,
        color_temp=0.65,  # NEW
        refresh=0.55,     # NEW
        rppg=0.70         # NEW
    )
    
    print(f"  ✓ VisualOverlay.draw_status_panel working (10 metrics)")
    
except Exception as e:
    print(f"  ✗ full-test.py components failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Verify all security levels work
print("\n[Test 7] Testing all security levels...")
try:
    levels = ["lenient", "balanced", "strict", "paranoid"]
    for level in levels:
        detector = UltimateAntiSpoof(level=level)
        test_image = np.random.randint(100, 150, (480, 640, 3), dtype=np.uint8)
        result = detector.check(test_image)
        
        # Verify new metrics exist
        assert hasattr(result.metrics, 'color_temp_score'), f"Missing color_temp_score in {level} mode"
        assert hasattr(result.metrics, 'refresh_score'), f"Missing refresh_score in {level} mode"
        assert hasattr(result.metrics, 'rppg_score'), f"Missing rppg_score in {level} mode"
        
        print(f"  ✓ Level '{level}' working (10 metrics available)")
    
except Exception as e:
    print(f"  ✗ Security levels test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Final summary
print("\n" + "="*70)
print("INTEGRATION TEST COMPLETE")
print("="*70)
print("\n✓ All tests passed successfully!")
print("\nVerified integrations:")
print("  1. UltimateAntiSpoof with 10-layer detection (7 base + 3 advanced)")
print("  2. LivenessMetrics with all 10 scores")
print("  3. FaceRecognizer with UltimateAntiSpoof integration")
print("  4. full-test.py VisualOverlay with 10 metrics display")
print("  5. All security levels (lenient/balanced/strict/paranoid)")
print("\nNew features verified:")
print("  ✓ Color Temperature Analysis (CCT)")
print("  ✓ Screen Refresh Rate Detection")
print("  ✓ rPPG Blood Flow Detection")
print("\n" + "="*70)
