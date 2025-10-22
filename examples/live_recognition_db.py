"""
Live Face Recognition with Database Integration

Real-time face detection and recognition with attendance database logging.

Usage:
    python examples/live_recognition_db.py --model models/recognizer.yml
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

# Fix Unicode encoding on Windows
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.face.detector import FaceDetector
from src.face.recognizer import FaceRecognizer
from attendance_database import AttendanceDatabase


def live_recognition_with_db(
    model_path: str,
    camera_index: int = 0,
    detection_mode: str = "highacc",
    show_confidence: bool = True,
    db_path: str = "attendance.db"
):
    """
    Run live face recognition with database logging.
    
    Args:
        model_path: Path to trained recognition model
        camera_index: Camera device index
        detection_mode: Detection mode (default, strict, highacc, ultra, live, extreme)
        show_confidence: Display confidence scores
        db_path: Path to attendance database
    """
    # Initialize database
    db = AttendanceDatabase(db_path)
    print(f"📊 Connected to database: {db_path}")
    
    # Load recognizer
    print("📥 Loading recognition model...")
    recognizer = FaceRecognizer()
    load_result = recognizer.load_model(model_path)
    
    print(f"✅ Model loaded successfully!")
    print(f"   • Model: {load_result['model_loaded']}")
    print(f"   • Total people: {load_result.get('total_labels', 'Unknown')}")
    
    # Get known people and ensure they're in database
    label_mapping = recognizer.get_label_mapping()
    for label_id, name in label_mapping.items():
        try:
            db.add_person(name, label_id)
            print(f"   • Added to DB: {name} (ID: {label_id})")
        except:
            pass  # Already exists
    
    # Initialize detector
    detector_params = {
        'default': {'scale_factor': 1.1, 'min_neighbors': 5, 'min_size': (30, 30)},
        'strict': {'scale_factor': 1.1, 'min_neighbors': 6, 'min_size': (50, 50)},
        'highacc': {'scale_factor': 1.05, 'min_neighbors': 7, 'min_size': (40, 40), 'use_clahe': True},
        'ultra': {'scale_factor': 1.03, 'min_neighbors': 8, 'min_size': (60, 60), 'use_clahe': True},
        'live': {'scale_factor': 1.05, 'min_neighbors': 9, 'min_size': (30, 30), 'use_clahe': True},
        'extreme': {'scale_factor': 1.02, 'min_neighbors': 12, 'min_size': (80, 80), 'use_clahe': True}
    }
    
    params = detector_params.get(detection_mode, detector_params['highacc'])
    print(f"\n🔍 Using detection mode: {detection_mode}")
    
    detector = FaceDetector(**params)
    
    # Open camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"❌ Error: Could not open camera {camera_index}")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print(f"\n🎥 Starting live recognition with database logging...")
    print("="*60)
    print("Controls:")
    print("  • 'q' - Quit")
    print("  • 's' - Save screenshot")
    print("  • 't' - Show today's attendance")
    print("  • 'a' - Show absentees")
    print("="*60 + "\n")
    
    # FPS calculation
    fps_start_time = time.time()
    fps_frame_count = 0
    fps_display = 0
    
    # Recently logged (to avoid duplicate logs in same session)
    recently_logged = {}
    log_cooldown = 5.0  # seconds
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Failed to read from camera")
                break
            
            # Detect faces
            faces = detector.detect_faces(frame)
            
            # Process each detected face
            for (x, y, w, h) in faces:
                # Extract face region
                face_region = frame[y:y+h, x:x+w]
                
                # Recognize face
                result = recognizer.predict_with_name(face_region)
                
                # Determine color and label based on recognition
                if result.recognized:
                    color = (0, 255, 0)  # Green for recognized
                    label = result.name
                    confidence = result.confidence
                    
                    # Log to database (with cooldown to avoid spam)
                    current_time = time.time()
                    last_log_time = recently_logged.get(label, 0)
                    
                    if (current_time - last_log_time) > log_cooldown:
                        if db.log_attendance(label, confidence, status="present"):
                            print(f"✓ Logged attendance: {label} (confidence: {confidence:.1f})")
                            recently_logged[label] = current_time
                else:
                    color = (0, 0, 255)  # Red for unknown
                    label = "Unknown"
                    confidence = result.confidence
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Prepare label text
                if show_confidence:
                    label_text = f"{label} ({confidence:.1f})"
                else:
                    label_text = label
                
                # Draw label background
                label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(
                    frame,
                    (x, y - label_size[1] - 10),
                    (x + label_size[0], y),
                    color,
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    frame,
                    label_text,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
            
            # Calculate FPS
            fps_frame_count += 1
            if time.time() - fps_start_time >= 1.0:
                fps_display = fps_frame_count
                fps_frame_count = 0
                fps_start_time = time.time()
            
            # Display FPS and info
            cv2.putText(
                frame,
                f"FPS: {fps_display}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            cv2.putText(
                frame,
                f"Faces: {len(faces)}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            # Display frame
            cv2.imshow('Live Face Recognition with Database', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                screenshot_dir = Path("screenshots")
                screenshot_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = screenshot_dir / f"recognition_{timestamp}.jpg"
                cv2.imwrite(str(screenshot_path), frame)
                print(f"📸 Screenshot saved: {screenshot_path}")
            elif key == ord('t'):
                # Show today's attendance
                print("\n" + "="*60)
                print("TODAY'S ATTENDANCE:")
                print("="*60)
                records = db.get_today_attendance()
                if records:
                    for record in records:
                        print(f"  {record['name']}: {record['timestamp']} (confidence: {record['confidence']:.1f})")
                else:
                    print("  No attendance records today")
                print("="*60 + "\n")
            elif key == ord('a'):
                # Show absentees
                print("\n" + "="*60)
                print("ABSENTEES TODAY:")
                print("="*60)
                absentees = db.get_absentees()
                if absentees:
                    for name in absentees:
                        print(f"  ❌ {name}")
                else:
                    print("  Everyone present!")
                print("="*60 + "\n")
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("✅ Live recognition ended")
    print("📊 Session Summary:")
    
    # Show today's attendance summary
    records = db.get_today_attendance()
    print(f"   • Total logged today: {len(records)}")
    
    absentees = db.get_absentees()
    print(f"   • Absentees: {len(absentees)}")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Live face recognition with database integration"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained recognition model (.yml)"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="highacc",
        choices=["default", "strict", "highacc", "ultra", "live", "extreme"],
        help="Detection mode (default: highacc)"
    )
    parser.add_argument(
        "--no-confidence",
        action="store_true",
        help="Hide confidence scores"
    )
    parser.add_argument(
        "--db",
        type=str,
        default="attendance.db",
        help="Path to attendance database (default: attendance.db)"
    )
    
    args = parser.parse_args()
    
    live_recognition_with_db(
        model_path=args.model,
        camera_index=args.camera,
        detection_mode=args.mode,
        show_confidence=not args.no_confidence,
        db_path=args.db
    )


if __name__ == "__main__":
    main()

