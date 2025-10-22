#!/usr/bin/env python3
"""
Debug Face Recognition - Shows detailed confidence and quality scores
"""

import sys
import cv2
from pathlib import Path

# Fix Windows terminal encoding
if sys.platform == "win32":
    try:
        import codecs
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, errors='replace')
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, errors='replace')
    except:
        pass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.face import FaceDetector, FaceRecognizer

def main():
    print("🔍 Debug Face Recognition")
    print("="*50)
    
    # Initialize components
    detector = FaceDetector()
    recognizer = FaceRecognizer(
        threshold=95.0,  # Distance-tolerant threshold
        use_enhanced_preprocessing=True
    )
    
    # Load model
    result = recognizer.load_model("models/clean_omar_model.yml")
    if not result['model_loaded']:
        print(f"❌ Failed to load model")
        return
    
    # Override the threshold after loading
    recognizer.set_threshold(95.0)  # Distance-tolerant threshold
    
    print(f"✅ Model loaded: {result['model_loaded']}")
    print(f"📊 Known people: {len(recognizer.get_known_people())}")
    print(f"🎯 Threshold: {recognizer.get_threshold()}")
    print(f"👥 Looking for: OMAR (OPTIMIZED MODEL - 514 SAMPLES - TARGET: Confidence 60)")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(f"❌ Failed to open camera")
        return
    
    print("\n🎥 Camera started. Press 'q' to quit")
    print("📋 Debug mode - showing detailed scores")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect faces
        faces = detector.detect_faces(frame)
        
        # Process each face
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            
            # Use standard recognition
            result = recognizer.predict_with_name(face_roi)
            
            # Debug output every 30 frames
            if frame_count % 30 == 0:
                # Calculate face size for distance info
                face_size = face_roi.shape[1] * face_roi.shape[0]  # width * height
                distance_category = "far" if face_size < 8000 else "close" if face_size > 20000 else "optimal"
                
                print(f"\n🔍 Frame {frame_count} Debug:")
                print(f"   Raw confidence: {result.raw_confidence:.1f}")
                print(f"   Smoothed confidence: {result.confidence:.1f}")
                print(f"   Threshold: {result.threshold:.1f}")
                print(f"   Quality score: {result.quality_score:.2f}")
                print(f"   Lighting: {result.lighting_metrics.condition.value}")
                print(f"   Face size: {face_size} pixels ({distance_category})")
                print(f"   Recognized: {result.recognized}")
                print(f"   Name: {result.name}")
                print(f"   Label ID: {result.label}")
            
            # Display results
            if result.recognized:
                color = (0, 255, 0)  # Green
                label = f"{result.name} ✅"
                conf_text = f"Conf: {result.confidence:.1f}"
                quality_text = f"Quality: {result.quality_score:.2f}"
            else:
                color = (0, 0, 255)  # Red
                label = "Unknown ❌"
                conf_text = f"Conf: {result.confidence:.1f}"
                quality_text = f"Quality: {result.quality_score:.2f}"
            
            # Draw rectangle and text
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, conf_text, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(frame, quality_text, (x, y+h+40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Show threshold info
            cv2.putText(frame, f"Threshold: {result.threshold:.1f}", (x, y+h+60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Show frame
        cv2.imshow("Debug Recognition", frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Debug session ended")

if __name__ == "__main__":
    main()
