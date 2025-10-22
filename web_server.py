#!/usr/bin/env python3
"""
Flask Web Server for Face Recognition Frontend

This server provides a web interface for the ultra-strict face recognition system.
It serves the frontend HTML and provides API endpoints for face recognition.
"""

import os
import sys
import base64
import io
import time
from pathlib import Path
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS

# Fix Windows terminal encoding
if sys.platform == "win32":
    try:
        import codecs
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, errors='replace')
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, errors='replace')
    except:
        pass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.face import FaceDetector, FaceRecognizer

class FaceRecognitionServer:
    def __init__(self):
        self.app = Flask(__name__, template_folder='frontend', static_folder='frontend')
        CORS(self.app)
        
        # Initialize face recognition components
        self.detector = FaceDetector()
        self.recognizer = FaceRecognizer(
            threshold=40.0,
            use_enhanced_preprocessing=True
        )
        
        # Load the trained model
        model_path = "models/combined_model.yml"
        if os.path.exists(model_path):
            result = self.recognizer.load_model(model_path)
            if result['model_loaded']:
                print(f"✅ Model loaded successfully!")
                print(f"   • Total people: {len(self.recognizer.get_known_people())}")
                print(f"   • Known people: {self.recognizer.get_known_people()}")
            else:
                print(f"❌ Failed to load model: {model_path}")
        else:
            print(f"❌ Model file not found: {model_path}")
        
        # Statistics
        self.stats = {
            'recognitions': 0,
            'rejections': 0,
            'total_frames': 0,
            'start_time': time.time()
        }
        
        self.setup_routes()
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Serve the main frontend page"""
            return send_from_directory('frontend', 'index.html')
        
        @self.app.route('/api/recognize', methods=['POST'])
        def recognize_face():
            """API endpoint for face recognition - processes all detected faces like ultra_strict_recognition.py"""
            try:
                # Get image data from request
                data = request.get_json()
                if not data or 'image' not in data:
                    return jsonify({'error': 'No image data provided'}), 400
                
                # Decode base64 image
                image_data = data['image'].split(',')[1]  # Remove data:image/jpeg;base64,
                image_bytes = base64.b64decode(image_data)
                
                # Convert to OpenCV format
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is None:
                    return jsonify({'error': 'Invalid image data'}), 400
                
                # Detect faces - like ultra_strict_recognition.py
                faces = self.detector.detect_faces(image)
                
                if len(faces) == 0:
                    return jsonify({
                        'faces': [],
                        'total_faces': 0,
                        'message': 'No faces detected'
                    })
                
                # Process each detected face - exactly like the original script
                face_results = []
                frame_recognitions = 0
                frame_rejections = 0
                
                for i, (x, y, w, h) in enumerate(faces):
                    # Extract face region
                    face_roi = image[y:y+h, x:x+w]
                    
                    # Perform ultra-strict recognition
                    result = self.recognizer.predict_with_name(face_roi)
                    
                    # Add face coordinates for frontend display
                    # Convert numpy types to Python native types for JSON serialization
                    face_result = {
                        'face_index': i,
                        'coordinates': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
                        'recognized': bool(result.recognized),
                        'name': str(result.name),
                        'confidence': float(result.confidence) if result.confidence != float('inf') else 999.0,
                        'quality': float(result.quality_score),
                        'threshold': float(result.threshold),
                        'raw_confidence': float(result.raw_confidence) if result.raw_confidence != float('inf') else 999.0
                    }
                    
                    face_results.append(face_result)
                    
                    # Count recognitions/rejections for this frame
                    if result.recognized:
                        frame_recognitions += 1
                    else:
                        frame_rejections += 1
                
                # Update statistics once per frame (like the original script)
                self.stats['total_frames'] += 1
                self.stats['recognitions'] += frame_recognitions
                self.stats['rejections'] += frame_rejections
                
                # Return all face results
                return jsonify({
                    'faces': face_results,
                    'total_faces': len(faces),
                    'frame_processed': True
                })
                
            except Exception as e:
                print(f"Recognition error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/stats', methods=['GET'])
        def get_stats():
            """Get recognition statistics"""
            elapsed = time.time() - self.stats['start_time']
            fps = self.stats['total_frames'] / elapsed if elapsed > 0 else 0
            
            total = self.stats['recognitions'] + self.stats['rejections']
            accuracy = (self.stats['recognitions'] / total * 100) if total > 0 else 0
            
            return jsonify({
                'recognitions': self.stats['recognitions'],
                'rejections': self.stats['rejections'],
                'total_frames': self.stats['total_frames'],
                'fps': round(fps, 1),
                'accuracy': round(accuracy, 1),
                'uptime': round(elapsed, 1)
            })
        
        @self.app.route('/api/model_info', methods=['GET'])
        def get_model_info():
            """Get information about the loaded model"""
            return jsonify({
                'model_loaded': self.recognizer.model is not None,
                'threshold': self.recognizer.threshold,
                'known_people': list(self.recognizer.label_to_name.values()) if hasattr(self.recognizer, 'label_to_name') else [],
                'total_labels': len(self.recognizer.label_to_name) if hasattr(self.recognizer, 'label_to_name') else 0
            })
        
        @self.app.route('/api/reset_stats', methods=['POST'])
        def reset_stats():
            """Reset recognition statistics"""
            self.stats = {
                'recognitions': 0,
                'rejections': 0,
                'total_frames': 0,
                'start_time': time.time()
            }
            return jsonify({'message': 'Statistics reset successfully'})
    
    def run(self, host='127.0.0.1', port=5000, debug=False):
        """Run the Flask server"""
        print("="*70)
        print("🌐 FACE RECOGNITION WEB SERVER")
        print("="*70)
        print(f"Server starting on: http://{host}:{port}")
        print(f"Frontend: http://{host}:{port}")
        print(f"API: http://{host}:{port}/api/")
        print("="*70)
        print("Press Ctrl+C to stop the server")
        print("="*70)
        
        self.app.run(host=host, port=port, debug=debug)

def main():
    """Main function to start the server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Face Recognition Web Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Create and run server
    server = FaceRecognitionServer()
    server.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
