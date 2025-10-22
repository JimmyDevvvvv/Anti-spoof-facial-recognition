"""
Train Recognition Model with Train-Test Split

Automatically splits data into 80% training and 20% testing for validation.

Usage:
    python examples/train_with_validation.py --data training_data --output models/recognizer.yml
    python examples/train_with_validation.py --data training_data --output models/recognizer.yml --split 0.2
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import random

import cv2
import numpy as np

# Fix Windows terminal encoding for emojis
if sys.platform == "win32":
    try:
        import codecs
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, errors='replace')
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, errors='replace')
    except Exception:
        pass  # If it fails, continue with default encoding

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.face.recognizer import FaceRecognizer


def load_and_split_data(
    data_dir: str, 
    test_split: float = 0.2,
    min_images_per_person: int = 5
) -> Tuple[List[np.ndarray], List[int], List[np.ndarray], List[int], Dict[int, str]]:
    """
    Load training data and split into train/test sets.
    
    Args:
        data_dir: Path to training data directory
        test_split: Fraction of data to use for testing (default: 0.2)
        min_images_per_person: Minimum images required per person (default: 5)
        
    Returns:
        tuple: (train_samples, train_labels, test_samples, test_labels, label_names)
        
    Raises:
        FileNotFoundError: If data directory doesn't exist
        ValueError: If no valid person directories found
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Training data directory not found: {data_dir}")
    
    # Get all person directories
    person_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    
    if not person_dirs:
        raise ValueError(f"No person directories found in {data_dir}")
    
    print(f"📂 Found {len(person_dirs)} person(s) in training data\n")
    
    train_samples = []
    train_labels = []
    test_samples = []
    test_labels = []
    label_names = {}
    
    skipped_persons = []
    
    for label_id, person_dir in enumerate(sorted(person_dirs)):
        person_name = person_dir.name.replace("_", " ")
        label_names[label_id] = person_name
        
        # Get all image files (support more formats)
        image_files = (
            list(person_dir.glob("*.jpg")) + 
            list(person_dir.glob("*.jpeg")) +
            list(person_dir.glob("*.png")) +
            list(person_dir.glob("*.JPG")) +
            list(person_dir.glob("*.JPEG")) +
            list(person_dir.glob("*.PNG"))
        )
        
        if not image_files:
            print(f"⚠️  Warning: No images found for {person_name}")
            skipped_persons.append(person_name)
            continue
        
        if len(image_files) < min_images_per_person:
            print(f"⚠️  Warning: {person_name} has only {len(image_files)} images "
                  f"(minimum recommended: {min_images_per_person})")
        
        # Shuffle images for random split
        random.shuffle(image_files)
        
        # Calculate split point (ensure at least 1 test sample if possible)
        if len(image_files) == 1:
            split_idx = 1
            test_files = []
        else:
            split_idx = max(1, int(len(image_files) * (1 - test_split)))
            test_files = image_files[split_idx:]
        
        train_files = image_files[:split_idx]
        
        print(f"👤 {person_name}:")
        print(f"   Total: {len(image_files)} images")
        print(f"   Train: {len(train_files)} images (ID: {label_id})")
        print(f"   Test:  {len(test_files)} images")
        
        # Load training images
        loaded_train = 0
        for img_path in train_files:
            try:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is not None and img.size > 0:
                    train_samples.append(img)
                    train_labels.append(label_id)
                    loaded_train += 1
                else:
                    print(f"   ⚠️  Failed to load: {img_path.name}")
            except Exception as e:
                print(f"   ⚠️  Error loading {img_path.name}: {e}")
        
        # Load test images
        loaded_test = 0
        for img_path in test_files:
            try:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is not None and img.size > 0:
                    test_samples.append(img)
                    test_labels.append(label_id)
                    loaded_test += 1
                else:
                    print(f"   ⚠️  Failed to load: {img_path.name}")
            except Exception as e:
                print(f"   ⚠️  Error loading {img_path.name}: {e}")
        
        if loaded_train != len(train_files) or loaded_test != len(test_files):
            print(f"   ✓ Successfully loaded: {loaded_train} train, {loaded_test} test")
        
        print()
    
    if skipped_persons:
        print(f"⚠️  Skipped {len(skipped_persons)} person(s) with no images: {', '.join(skipped_persons)}\n")
    
    print(f"📊 Dataset Summary:")
    print(f"   Training samples: {len(train_samples)}")
    print(f"   Test samples: {len(test_samples)}")
    print(f"   Total people: {len(label_names)}")
    
    if len(train_samples) == 0:
        raise ValueError("No training samples were successfully loaded!")
    
    return train_samples, train_labels, test_samples, test_labels, label_names


def evaluate_model(
    recognizer: FaceRecognizer,
    test_samples: List[np.ndarray],
    test_labels: List[int],
    label_names: Dict[int, str]
) -> Dict:
    """
    Evaluate model on test set.
    
    Args:
        recognizer: Trained FaceRecognizer
        test_samples: Test face images
        test_labels: True labels for test samples
        label_names: Mapping of label IDs to names
        
    Returns:
        dict: Evaluation metrics including accuracy, FAR, FRR, and per-person accuracy
    """
    print("\n" + "="*60)
    print("EVALUATING MODEL ON TEST SET")
    print("="*60 + "\n")
    
    if len(test_samples) == 0:
        print("⚠️  No test samples available for evaluation")
        return {}
    
    correct = 0
    total = len(test_samples)
    
    predictions = []
    confidences = []
    
    # Per-person metrics
    person_correct = {label: 0 for label in set(test_labels)}
    person_total = {label: 0 for label in set(test_labels)}
    
    for i, (sample, true_label) in enumerate(zip(test_samples, test_labels)):
        # Predict
        try:
            pred_label, confidence = recognizer.predict(sample, return_confidence=True)
        except Exception as e:
            print(f"⚠️  Error predicting sample {i+1}: {e}")
            pred_label = -1
            confidence = 0.0
        
        # Convert name back to label if needed
        if isinstance(pred_label, str):
            pred_label = next((k for k, v in label_names.items() if v == pred_label), -1)
        
        predictions.append(pred_label)
        confidences.append(confidence)
        
        # Check if correct
        is_correct = pred_label == true_label
        if is_correct:
            correct += 1
            person_correct[true_label] += 1
        
        person_total[true_label] += 1
        
        # Print individual result
        true_name = label_names.get(true_label, f"ID_{true_label}")
        pred_name = label_names.get(pred_label, "Unknown") if pred_label != -1 else "Unknown"
        status = "✓" if is_correct else "✗"
        
        print(f"{status} Test {i+1}/{total}: True={true_name:15} Pred={pred_name:15} Conf={confidence:5.1f}")
    
    # Calculate metrics
    accuracy = (correct / total * 100) if total > 0 else 0
    
    # False acceptance and rejection rates
    false_positives = sum(1 for pred, true in zip(predictions, test_labels) 
                         if pred != true and pred != -1)
    false_negatives = sum(1 for pred, true in zip(predictions, test_labels) 
                          if pred == -1 and true != -1)
    
    far = (false_positives / total * 100) if total > 0 else 0
    frr = (false_negatives / total * 100) if total > 0 else 0
    
    # Print overall results
    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    print(f"Total Test Samples:     {total}")
    print(f"Correct Predictions:    {correct}")
    print(f"Incorrect Predictions:  {total - correct}")
    print()
    print(f"Accuracy:               {accuracy:.2f}% (goal: ≥90%)")
    print(f"False Acceptance Rate:  {far:.2f}% (goal: ≤2%)")
    print(f"False Rejection Rate:   {frr:.2f}% (goal: ≤5%)")
    print()
    
    if confidences:
        print(f"Average Confidence:     {np.mean(confidences):.2f}")
        print(f"Confidence Range:       {np.min(confidences):.2f} - {np.max(confidences):.2f}")
    
    # Per-person accuracy
    print("\n" + "-"*60)
    print("PER-PERSON ACCURACY")
    print("-"*60)
    for label in sorted(person_total.keys()):
        name = label_names.get(label, f"ID_{label}")
        person_acc = (person_correct[label] / person_total[label] * 100) if person_total[label] > 0 else 0
        print(f"{name:20} {person_correct[label]:2}/{person_total[label]:2} ({person_acc:5.1f}%)")
    
    print("="*60)
    
    # Pass/Fail with detailed feedback
    if accuracy >= 90 and far <= 2 and frr <= 5:
        print("\n✅ MODEL PASSED VALIDATION!")
        print("   All metrics meet target thresholds")
    elif accuracy >= 90:
        print("\n✅ MODEL PASSED ACCURACY TARGET!")
        if far > 2:
            print(f"   ⚠️  High False Acceptance Rate: {far:.2f}% (target: ≤2%)")
        if frr > 5:
            print(f"   ⚠️  High False Rejection Rate: {frr:.2f}% (target: ≤5%)")
    else:
        print("\n⚠️  MODEL NEEDS IMPROVEMENT")
        print(f"   Accuracy is {accuracy:.1f}% (target: ≥90%)")
        print("\nSuggestions:")
        print("  • Capture more training samples per person (recommended: 20-50)")
        print("  • Ensure good quality images (proper lighting, clear focus, frontal faces)")
        print("  • Check for consistent capture conditions across all images")
        print("  • Consider adjusting LBPH parameters (--radius, --neighbors, --threshold)")
    
    return {
        'accuracy': accuracy,
        'false_acceptance_rate': far,
        'false_rejection_rate': frr,
        'total_samples': total,
        'correct': correct,
        'confidences': confidences,
        'per_person_accuracy': {
            label_names[label]: (person_correct[label] / person_total[label] * 100) 
            if person_total[label] > 0 else 0
            for label in person_total.keys()
        }
    }


def train_with_validation(
    data_dir: str,
    output_path: str,
    test_split: float = 0.2,
    radius: int = 1,
    neighbors: int = 8,
    grid_x: int = 8,
    grid_y: int = 8,
    threshold: float = 50.0,
    use_enhanced_preprocessing: bool = True
) -> Tuple[Optional[FaceRecognizer], Optional[Dict]]:
    """
    Train model with train-test split validation.
    
    Args:
        data_dir: Path to training data directory
        output_path: Path to save trained model
        test_split: Fraction of data for testing
        radius: LBPH radius parameter
        neighbors: LBPH neighbors parameter
        grid_x: LBPH grid_x parameter
        grid_y: LBPH grid_y parameter
        threshold: Recognition confidence threshold
        use_enhanced_preprocessing: Whether to use enhanced preprocessing
        
    Returns:
        tuple: (recognizer, metrics) or (None, None) on failure
    """
    print("="*60)
    print("🤖 FACE RECOGNITION TRAINING WITH VALIDATION")
    print("="*60 + "\n")
    
    try:
        # Load and split data
        train_samples, train_labels, test_samples, test_labels, label_names = load_and_split_data(
            data_dir, test_split
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ Error loading data: {e}")
        return None, None
    
    if len(train_samples) == 0:
        print("❌ Error: No training samples found!")
        return None, None
    
    # Create output directory if it doesn't exist
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Create recognizer
    print(f"\n🔧 Creating LBPH recognizer with parameters:")
    print(f"   • Radius: {radius}")
    print(f"   • Neighbors: {neighbors}")
    print(f"   • Grid: {grid_x}x{grid_y}")
    print(f"   • Threshold: {threshold}")
    print(f"   • Enhanced preprocessing: {'Enabled' if use_enhanced_preprocessing else 'Disabled (basic only)'}")
    
    recognizer = FaceRecognizer(
        radius=radius,
        neighbors=neighbors,
        grid_x=grid_x,
        grid_y=grid_y,
        threshold=threshold,
        use_enhanced_preprocessing=use_enhanced_preprocessing
    )
    
    # Train model
    print(f"\n🎓 Training model on {len(train_samples)} samples...")
    try:
        results = recognizer.train(train_samples, train_labels, label_names)
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return None, None
    
    print(f"\n✅ Training completed!")
    print(f"   • Samples processed: {results['samples_processed']}")
    print(f"   • Unique labels: {results['unique_labels']}")
    
    # Evaluate on test set
    metrics = None
    if len(test_samples) > 0:
        metrics = evaluate_model(recognizer, test_samples, test_labels, label_names)
    else:
        print("\n⚠️  No test samples available for validation")
        print("   Consider increasing the number of images per person")
    
    # Save model
    print(f"\n💾 Saving model to: {output_path}")
    try:
        save_results = recognizer.save_model(output_path)
        print(f"   ✓ Model saved: {save_results['model_path']}")
        print(f"   ✓ Metadata saved: {save_results['metadata_path']}")
    except Exception as e:
        print(f"   ❌ Error saving model: {e}")
        return recognizer, metrics
    
    # Display label mappings
    print(f"\n📋 Label Mappings:")
    for label_id, name in sorted(label_names.items()):
        print(f"   {label_id}: {name}")
    
    print(f"\n" + "="*60)
    print("✅ Training and validation complete!")
    if metrics and metrics.get('accuracy', 0) >= 90:
        print("🎉 Model is ready for deployment!")
    elif metrics:
        print("⚠️  Consider capturing more training data to improve accuracy")
    print("="*60)
    
    return recognizer, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train face recognition model with train-test split validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data",
        type=str,
        default="training_data",
        help="Path to training data directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/recognizer.yml",
        help="Path to save trained model"
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.2,
        help="Test set split ratio (0.0-0.5)"
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=1,
        help="LBPH radius parameter"
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=8,
        help="LBPH neighbors parameter"
    )
    parser.add_argument(
        "--grid-x",
        type=int,
        default=8,
        help="LBPH grid_x parameter"
    )
    parser.add_argument(
        "--grid-y",
        type=int,
        default=8,
        help="LBPH grid_y parameter"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=50.0,
        help="Recognition confidence threshold"
    )
    parser.add_argument(
        "--no-preprocessing",
        action="store_true",
        help="Disable enhanced preprocessing (use basic preprocessing only)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not 0.0 <= args.split <= 0.5:
        parser.error("--split must be between 0.0 and 0.5")
    
    if args.radius < 1:
        parser.error("--radius must be at least 1")
    
    if args.neighbors < 4:
        parser.error("--neighbors must be at least 4")
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    recognizer, metrics = train_with_validation(
        data_dir=args.data,
        output_path=args.output,
        test_split=args.split,
        radius=args.radius,
        neighbors=args.neighbors,
        grid_x=args.grid_x,
        grid_y=args.grid_y,
        threshold=args.threshold,
        use_enhanced_preprocessing=not args.no_preprocessing
    )
    
    # Exit with appropriate code
    if recognizer is None:
        sys.exit(1)
    elif metrics and metrics.get('accuracy', 0) < 90:
        sys.exit(2)  # Training succeeded but model needs improvement
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()