"""Simple face recognition model training - no quality filtering."""

import sys
from pathlib import Path
import cv2
import numpy as np
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.face.recognizer import FaceRecognizer

def load_training_data(data_dir: str) -> Tuple[List[np.ndarray], List[int], Dict[int, str]]:
    """Load training data without quality filtering."""
    data_path = Path(data_dir)
    samples = []
    labels = []
    label_names = {}
    
    # Filter out backup folders
    person_dirs = sorted([d for d in data_path.iterdir() 
                         if d.is_dir() and 'backup' not in d.name.lower()])
    
    print(f"📂 Found {len(person_dirs)} people\n")
    
    for label_id, person_dir in enumerate(person_dirs):
        person_name = person_dir.name.replace("_", " ")
        label_names[label_id] = person_name
        
        # Get all images
        image_files = (
            list(person_dir.glob("*.jpg")) + 
            list(person_dir.glob("*.jpeg")) +
            list(person_dir.glob("*.png"))
        )
        
        if not image_files:
            continue
        
        person_samples = []
        for img_file in image_files:
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                person_samples.append(img)
                labels.append(label_id)
        
        samples.extend(person_samples)
        print(f"👤 {person_name}: {len(person_samples)} images (ID: {label_id})")
    
    print(f"\n📊 Total: {len(samples)} images from {len(label_names)} people\n")
    return samples, labels, label_names


def main():
    """Train the model."""
    data_dir = "Data/combined_dataset"
    output_model = "models/combined_model_retrained.yml"
    
    print("="*60)
    print("SIMPLE FACE RECOGNITION TRAINING")
    print("="*60 + "\n")
    
    # Load data
    samples, labels, label_names = load_training_data(data_dir)
    
    if not samples:
        print("❌ No training data found!")
        return 1
    
    # Create recognizer WITHOUT enhanced preprocessing
    print("🔧 Creating LBPH recognizer...")
    recognizer = FaceRecognizer(
        threshold=55.0,  # Set to 55 as requested
        use_enhanced_preprocessing=False,  # DISABLE strict quality checks
        enable_antispoofing=False  # Disable for training
    )
    
    # Train
    print("\n🎓 Training model...")
    try:
        recognizer.train(samples, labels, label_names)
        print("✓ Training complete!\n")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return 1
    
    # Save
    print(f"💾 Saving model to {output_model}...")
    recognizer.save_model(output_model)
    print("✓ Model saved!\n")
    
    print("="*60)
    print("✓ TRAINING COMPLETE!")
    print("="*60)
    print(f"\nModel file: {output_model}")
    print(f"People trained: {len(label_names)}")
    print(f"Total images: {len(samples)}\n")
    
    # Show warning about OMAR imbalance
    omar_count = sum(1 for l in labels if label_names[l] == 'OMAR')
    avg_count = len(samples) / len(label_names)
    if omar_count > avg_count * 2:
        print("⚠️  WARNING: Dataset is imbalanced!")
        print(f"   OMAR has {omar_count} images ({omar_count/len(samples)*100:.1f}% of total)")
        print(f"   Average per person: {avg_count:.0f} images")
        print(f"   This may cause bias toward recognizing everyone as OMAR.")
        print(f"\n💡 Recommendation: Balance the dataset by:")
        print(f"   1. Reducing OMAR images to ~50-100")
        print(f"   2. Adding more images for other people")
        print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
