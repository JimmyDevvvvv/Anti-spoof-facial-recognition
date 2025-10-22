#!/usr/bin/env python3
"""
Train Model on Combined Dataset

This script trains a face recognition model using the combined dataset.

Usage:
    python train_final_omar_model.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from examples.train_with_validation import train_with_validation


def main():
    print("="*70)
    print("🤖 TRAINING MODEL ON COMBINED DATASET")
    print("="*70)
    
    # Train on combined dataset (no split needed - already split)
    recognizer, metrics = train_with_validation(
        data_dir="final_omar_dataset/train",
        output_path="models/final_omar_model.yml",
        test_split=0.0,  # No split needed, we already have separate test set
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8,
        threshold=50.0,
        use_enhanced_preprocessing=True
    )
    
    if recognizer is None:
        print("\n❌ Training failed!")
        return 1
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Evaluate on test set:")
    print("   python evaluate_final_omar_model.py")
    print("\n2. Test with ultra-strict recognition:")
    print("   python examples/ultra_strict_recognition.py --model models/final_omar_model.yml")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
