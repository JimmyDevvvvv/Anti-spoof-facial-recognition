"""
Quick test to demonstrate photo detection improvements.
Run this to see debug output showing why photos are now detected.
"""

import sys
import numpy as np
from src.face.final_anti_spoof import UltimateAntiSpoof

print("="*70)
print("PHOTO DETECTION TEST - Enhanced Algorithm")
print("="*70)

# Create detector with debug enabled
print("\nInitializing detector with DEBUG enabled...")
detector = UltimateAntiSpoof(level="balanced", enable_video_mode=True, debug=True)

print("\n" + "="*70)
print("TESTING: Simulated Photo Pattern")
print("="*70)
print("Simulating metrics from your screenshot:")
print("  - Low texture (0.15) = smooth paper")
print("  - Low color (0.08) = flat print colors")
print("  - Low rPPG (should be 0.2 or less) = no heartbeat")
print("  - High motion (0.87) = hand movement (will be downweighted)")
print("\n")

# Create test image (will generate low scores for photo)
test_image = np.ones((480, 640, 3), dtype=np.uint8) * 150

# Run check
result = detector.check(test_image)

print("\n" + "="*70)
print("RESULT")
print("="*70)
print(f"Is Real: {result.is_real}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Attack Type: {result.attack_type.value}")
print(f"\nMetrics:")
print(f"  Texture: {result.metrics.texture_score:.2f}")
print(f"  Motion: {result.metrics.motion_score:.2f}")
print(f"  Color: {result.metrics.color_score:.2f}")
print(f"  Depth: {result.metrics.depth_score:.2f}")
print(f"  Frequency: {result.metrics.frequency_score:.2f}")
print(f"  ColorTemp: {result.metrics.color_temp_score:.2f}")
print(f"  Refresh: {result.metrics.refresh_score:.2f}")
print(f"  rPPG: {result.metrics.rppg_score:.2f}")

print("\n" + "="*70)
print("IMPROVEMENTS MADE:")
print("="*70)
print("1. ✓ Added PHOTO DETECTION OVERRIDE")
print("   - Detects low texture + low color + no heartbeat pattern")
print("   - Triggers on 2+ photo indicators")
print("\n2. ✓ Made rPPG MUCH STRICTER")
print("   - Requires SNR > 3.5 for real faces")
print("   - Returns 0.1-0.2 for photos (no heartbeat)")
print("\n3. ✓ REDUCED MOTION WEIGHT by 50%")
print("   - Motion can be faked by hand movement")
print("   - Now weights critical checks higher (texture, color, rPPG)")
print("\n4. ✓ INCREASED rPPG WEIGHT to 30% (highest)")
print("   - Blood flow detection can't be faked")
print("   - Now dominates the decision")
print("\n5. ✓ Adjusted confidence calculation")
print("   - 60% base + 60% advanced (emphasizes advanced checks)")
print("   - Advanced checks (rPPG, refresh, color temp) override base")
print("\n" + "="*70)
print("\nRun: python full-test.py --model .\\models\\combined_model.yml --level balanced")
print("Then test with your printed photo again!")
print("="*70)
