#!/usr/bin/env python3
"""
Start Face Recognition Web Server

This script starts the web server for the face recognition frontend.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    try:
        import flask
        import flask_cors
        import cv2
        import numpy
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def main():
    """Main function"""
    print("="*70)
    print("🚀 STARTING FACE RECOGNITION WEB SERVER")
    print("="*70)
    
    # Check requirements
    if not check_requirements():
        return 1
    
    # Check if model exists
    model_path = Path("models/combined_model.yml")
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        print("Please train a model first using the training scripts")
        return 1
    
    print(f"✅ Model found: {model_path}")
    
    # Start the web server
    print("\n🌐 Starting web server...")
    print("The frontend will be available at: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server")
    print("="*70)
    
    try:
        # Run the web server
        subprocess.run([sys.executable, "web_server.py"], check=True)
    except KeyboardInterrupt:
        print("\n✅ Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Server error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
