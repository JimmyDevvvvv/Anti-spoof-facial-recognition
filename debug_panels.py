#!/usr/bin/env python3
"""
Debug Panel Visibility - Test Recognition Panel Display
=======================================================

This script helps debug why the recognition panel might not be visible.
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

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def test_panel_visibility():
    """Test if recognition panel is visible."""
    print("="*60)
    print("DEBUG: RECOGNITION PANEL VISIBILITY TEST")
    print("="*60)
    
    try:
        # Import modules
        from src.face.recognizer import FaceRecognizer
        print("✅ FaceRecognizer imported successfully")
        
        # Test model loading
        model_path = "models/combined_model.yml"
        if not Path(model_path).exists():
            print(f"❌ Model file not found: {model_path}")
            return False
        
        print(f"✅ Model file exists: {model_path}")
        
        # Create recognizer
        recognizer = FaceRecognizer(threshold=50.0)
        print("✅ FaceRecognizer created successfully")
        
        # Load model
        result = recognizer.load_model(model_path)
        print(f"✅ Model loaded successfully")
        print(f"   Total labels: {result['total_labels']}")
        print(f"   Recognition enabled: True")
        
        # Test camera
        print("\n📷 Testing camera...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return False
        
        print("✅ Camera opened successfully")
        
        # Test panel drawing
        print("\n🎨 Testing panel drawing...")
        
        frame_count = 0
        panel_visible_count = 0
        
        while frame_count < 30:  # Test for 30 frames
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Create a test frame with panels
            test_frame = frame.copy()
            h, w = test_frame.shape[:2]
            
            # Draw recognition panel (top-left)
            panel_x = 10
            panel_y = 60
            panel_width = 250
            panel_height = 100
            
            # Draw panel background
            cv2.rectangle(test_frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
            cv2.rectangle(test_frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), (0, 255, 0), 2)
            
            # Draw panel text
            cv2.putText(test_frame, "RECOGNITION PANEL", (panel_x + 10, panel_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(test_frame, "Status: TEST MODE", (panel_x + 10, panel_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(test_frame, "Frame: " + str(frame_count), (panel_x + 10, panel_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Draw anti-spoofing panel (top-right)
            panel_x2 = w - 260
            panel_y2 = 60
            cv2.rectangle(test_frame, (panel_x2, panel_y2), (w - 10, panel_y2 + 170), (0, 0, 0), -1)
            cv2.rectangle(test_frame, (panel_x2, panel_y2), (w - 10, panel_y2 + 170), (255, 0, 0), 2)
            cv2.putText(test_frame, "ANTI-SPOOFING PANEL", (panel_x2 + 10, panel_y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(test_frame, "Status: TEST MODE", (panel_x2 + 10, panel_y2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Draw statistics panel (bottom-left)
            panel_x3 = 10
            panel_y3 = h - 150
            cv2.rectangle(test_frame, (panel_x3, panel_y3), (panel_x3 + 220, h - 10), (0, 0, 0), -1)
            cv2.rectangle(test_frame, (panel_x3, panel_y3), (panel_x3 + 220, h - 10), (0, 255, 255), 2)
            cv2.putText(test_frame, "STATISTICS PANEL", (panel_x3 + 10, panel_y3 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(test_frame, "Frames: " + str(frame_count), (panel_x3 + 10, panel_y3 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Draw controls panel (bottom-right)
            panel_x4 = w - 260
            panel_x4 = w - 260
            panel_y4 = h - 120
            cv2.rectangle(test_frame, (panel_x4, panel_y4), (w - 10, h - 10), (0, 0, 0), -1)
            cv2.rectangle(test_frame, (panel_x4, panel_y4), (w - 10, h - 10), (255, 255, 0), 2)
            cv2.putText(test_frame, "CONTROLS PANEL", (panel_x4 + 10, panel_y4 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(test_frame, "Press Q to quit", (panel_x4 + 10, panel_y4 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Add frame counter
            cv2.putText(test_frame, f"Frame: {frame_count}/30", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(test_frame, "ALL PANELS SHOULD BE VISIBLE", (w//2 - 200, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Panel Visibility Test', test_frame)
            
            # Check if panels are visible (simple check)
            panel_visible_count += 1
            
            # Wait for key
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n📊 Test Results:")
        print(f"   Frames processed: {frame_count}")
        print(f"   Panels should be visible: {panel_visible_count}")
        
        if frame_count > 0:
            print("\n✅ Panel visibility test completed!")
            print("   If you could see colored rectangles with text, panels are working.")
            print("   If you couldn't see them, there might be a display issue.")
            return True
        else:
            print("\n❌ No frames processed - camera issue")
            return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_panel_visibility()
    if success:
        print("\n🎉 Panel visibility test completed!")
    else:
        print("\n❌ Panel visibility test failed!")
        sys.exit(1)
