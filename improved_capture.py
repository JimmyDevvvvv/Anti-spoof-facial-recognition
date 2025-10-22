#!/usr/bin/env python3
"""
Improved Face Capture with Better Quality Settings

This script improves the capture quality by:
- Higher resolution (1080p)
- Better face detection parameters
- Full face capture validation
- Improved quality thresholds
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Fix Windows terminal encoding
if sys.platform == "win32":
    try:
        import codecs
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, errors='replace')
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, errors='replace')
    except Exception:
        pass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.face import FaceDetector, FacePreprocessor

def capture_high_quality_faces(person_name: str, num_samples: int = 50):
    """Capture high-quality face images with full face validation."""
    
    print("🎥 High-Quality Face Capture")
    print("="*50)
    print(f"Person: {person_name}")
    print(f"Samples: {num_samples}")
    print("="*50)
    
    # Initialize camera with HIGHER resolution
    cap = cv2.VideoCapture(0)
    
    # Set HIGHER resolution for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)   # 1080p width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # 1080p height
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Enable auto-focus and auto-exposure
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    
    # Initialize detector with BETTER parameters for full face capture
    detector = FaceDetector(
        scale_factor=1.1,        # Slightly less conservative
        min_neighbors=5,         # Lower for better detection
        min_size=(100, 100),    # LARGER minimum size for full faces
        use_clahe=True
    )
    
    # Initialize preprocessor
    preprocessor = FacePreprocessor(
        target_size=(150, 150),  # LARGER target size for better quality
        use_clahe=True,
        clahe_clip_limit=2.0
    )
    
    # Create output directory
    output_dir = Path("training_data") / person_name.replace(" ", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    captured_count = 0
    frame_count = 0
    
    print("\n📋 Instructions:")
    print("• Position your FULL FACE in the center of the frame")
    print("• Ensure your chin is visible")
    print("• Move slowly and naturally")
    print("• Press SPACE to capture")
    print("• Press 'q' to quit")
    print("\n🎥 Starting capture...")
    
    while captured_count < num_samples:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect faces
        faces = detector.detect_faces(frame)
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Check if face is large enough (full face validation)
            face_area = w * h
            frame_area = frame.shape[0] * frame.shape[1]
            face_ratio = face_area / frame_area
            
            # Validate face size (should be 5-25% of frame for good quality)
            if 0.05 <= face_ratio <= 0.25:
                cv2.putText(frame, "GOOD SIZE", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "TOO SMALL/LARGE", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show face dimensions
            cv2.putText(frame, f"Size: {w}x{h}", (x, y+h+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Ratio: {face_ratio:.1%}", (x, y+h+40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show capture progress
        cv2.putText(frame, f"Captured: {captured_count}/{num_samples}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Resolution: {frame.shape[1]}x{frame.shape[0]}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow("High-Quality Face Capture", frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space bar to capture
            if len(faces) == 1:
                x, y, w, h = faces[0]
                face_roi = frame[y:y+h, x:x+w]
                
                # Validate face size
                face_area = w * h
                frame_area = frame.shape[0] * frame.shape[1]
                face_ratio = face_area / frame_area
                
                if 0.05 <= face_ratio <= 0.25:  # Good size range
                    # Preprocess face
                    processed_face = preprocessor.preprocess(face_roi)
                    
                    # Save image
                    filename = f"{person_name}_{captured_count:03d}_1080p.jpg"
                    filepath = output_dir / filename
                    
                    success = cv2.imwrite(str(filepath), processed_face)
                    
                    if success:
                        captured_count += 1
                        print(f"✅ Captured {captured_count}/{num_samples}: {filename}")
                        print(f"   Face size: {w}x{h} ({face_ratio:.1%} of frame)")
                    else:
                        print(f"❌ Failed to save: {filename}")
                else:
                    print(f"⚠️  Face too {'small' if face_ratio < 0.05 else 'large'}: {face_ratio:.1%}")
            else:
                print(f"⚠️  Need exactly 1 face, detected: {len(faces)}")
        
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Capture complete!")
    print(f"📊 Captured: {captured_count}/{num_samples} images")
    print(f"📁 Saved to: {output_dir}")
    print(f"🎯 Resolution: 1080p (1920x1080)")
    print(f"📏 Face size: 150x150 pixels")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="High-quality face capture")
    parser.add_argument("--name", required=True, help="Person's name")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples")
    
    args = parser.parse_args()
    
    capture_high_quality_faces(args.name, args.samples)
