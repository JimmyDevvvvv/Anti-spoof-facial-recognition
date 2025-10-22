#!/usr/bin/env python3
"""
Evaluate Combined Model on Test Set

This script evaluates the trained model on the held-out test set.

Usage:
    python evaluate_final_omar_model.py
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.face.recognizer import FaceRecognizer


def load_test_data(test_dir: Path) -> Tuple[List[np.ndarray], List[int], Dict[int, str]]:
    """Load test images and labels."""
    test_images = []
    test_labels = []
    label_names = {}
    
    person_dirs = sorted([d for d in test_dir.iterdir() if d.is_dir()])
    
    print(f"📂 Loading test data from {len(person_dirs)} people...\n")
    
    for label_id, person_dir in enumerate(person_dirs):
        person_name = person_dir.name
        label_names[label_id] = person_name
        
        image_files = list(person_dir.glob("*.jpg"))
        
        for img_file in image_files:
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                test_images.append(img)
                test_labels.append(label_id)
        
        print(f"   ✓ {person_name}: {len([f for f in image_files if cv2.imread(str(f)) is not None])} test images")
    
    return test_images, test_labels, label_names


def evaluate_model(
    recognizer: FaceRecognizer,
    test_images: List[np.ndarray],
    test_labels: List[int],
    label_names: Dict[int, str]
):
    """Evaluate model on test set."""
    
    print("\n" + "="*70)
    print("🔍 EVALUATING MODEL ON TEST SET")
    print("="*70 + "\n")
    
    correct = 0
    total = len(test_images)
    confidences = []
    
    # Per-person metrics
    person_correct = {label: 0 for label in set(test_labels)}
    person_total = {label: 0 for label in set(test_labels)}
    
    for i, (img, true_label) in enumerate(zip(test_images, test_labels)):
        try:
            pred_label, confidence = recognizer.predict(img, return_confidence=True)
        except Exception as e:
            print(f"⚠️  Error predicting sample {i+1}: {e}")
            pred_label = -1
            confidence = 0.0
        
        # Convert name back to label
        if isinstance(pred_label, str):
            pred_label = next((k for k, v in label_names.items() if v == pred_label), -1)
        
        confidences.append(confidence)
        
        is_correct = pred_label == true_label
        if is_correct:
            correct += 1
            person_correct[true_label] += 1
        
        person_total[true_label] += 1
        
        true_name = label_names.get(true_label, f"ID_{true_label}")
        pred_name = label_names.get(pred_label, "Unknown") if pred_label != -1 else "Unknown"
        status = "✓" if is_correct else "✗"
        
        if i < 20 or not is_correct:  # Show first 20 and all errors
            print(f"{status} Test {i+1}/{total}: {true_name:20} → {pred_name:20} (conf: {confidence:5.1f})")
        elif i == 20:
            print(f"   ... (showing errors only from here) ...")
    
    # Calculate metrics
    accuracy = (correct / total * 100) if total > 0 else 0
    
    false_positives = sum(1 for pred, true in zip(
        [recognizer.predict(img, return_confidence=True)[0] for img in test_images],
        test_labels
    ) if pred != true and pred != -1)
    
    false_negatives = sum(1 for pred in [
        recognizer.predict(img, return_confidence=True)[0] for img in test_images
    ] if pred == -1)
    
    far = (false_positives / total * 100) if total > 0 else 0
    frr = (false_negatives / total * 100) if total > 0 else 0
    
    # Print results
    print("\n" + "="*70)
    print("📊 EVALUATION RESULTS")
    print("="*70)
    print(f"Total Test Images:      {total}")
    print(f"Correct Predictions:    {correct} ({accuracy:.2f}%)")
    print(f"Incorrect Predictions:  {total - correct} ({100-accuracy:.2f}%)")
    print()
    print(f"False Acceptance Rate:  {far:.2f}% (goal: ≤2%)")
    print(f"False Rejection Rate:   {frr:.2f}% (goal: ≤5%)")
    print()
    print(f"Average Confidence:     {np.mean(confidences):.2f}")
    print(f"Confidence Std Dev:     {np.std(confidences):.2f}")
    
    # Per-person accuracy
    print("\n" + "-"*70)
    print("PER-PERSON ACCURACY (Top 10 and Bottom 10)")
    print("-"*70)
    
    person_accuracies = {
        label: (person_correct[label] / person_total[label] * 100) if person_total[label] > 0 else 0
        for label in person_total.keys()
    }
    
    sorted_accuracies = sorted(person_accuracies.items(), key=lambda x: x[1], reverse=True)
    
    # Top 10
    for label, acc in sorted_accuracies[:10]:
        name = label_names.get(label, f"ID_{label}")
        print(f"✅ {name:20} {person_correct[label]:3}/{person_total[label]:3} ({acc:5.1f}%)")
    
    if len(sorted_accuracies) > 20:
        print("   ...")
    
    # Bottom 10
    for label, acc in sorted_accuracies[-10:]:
        name = label_names.get(label, f"ID_{label}")
        print(f"⚠️  {name:20} {person_correct[label]:3}/{person_total[label]:3} ({acc:5.1f}%)")
    
    print("="*70)
    
    # Final verdict
    if accuracy >= 90 and far <= 2 and frr <= 5:
        print("\n✅ MODEL PASSED ALL EVALUATION CRITERIA!")
        print("   Ready for deployment!")
    elif accuracy >= 90:
        print("\n✅ MODEL PASSED ACCURACY TARGET!")
        print("   ⚠️  But check FAR/FRR values")
    else:
        print("\n⚠️  MODEL NEEDS IMPROVEMENT")
        print("   Consider capturing more/better training data")
    
    print("="*70)
    
    return accuracy >= 90


def main():
    print("="*70)
    print("📊 EVALUATING COMBINED MODEL ON TEST SET")
    print("="*70)
    
    # Load model
    model_path = Path("models/final_omar_model.yml")
    if not model_path.exists():
        print(f"\n❌ Model not found: {model_path}")
        print("   Run train_final_omar_model.py first!")
        return 1
    
    recognizer = FaceRecognizer(threshold=50.0, use_enhanced_preprocessing=True)
    result = recognizer.load_model(str(model_path))
    
    if not result.get('model_loaded'):
        print("\n❌ Failed to load model!")
        return 1
    
    print(f"\n✅ Model loaded successfully")
    print(f"📋 Known people: {len(result.get('label_mappings', {}))}")
    
    # Load test data
    test_dir = Path("final_omar_dataset/test")
    if not test_dir.exists():
        print(f"\n❌ Test directory not found: {test_dir}")
        return 1
    
    test_images, test_labels, label_names = load_test_data(test_dir)
    
    if not test_images:
        print("\n❌ No test images found!")
        return 1
    
    # Evaluate
    passed = evaluate_model(recognizer, test_images, test_labels, label_names)
    
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
