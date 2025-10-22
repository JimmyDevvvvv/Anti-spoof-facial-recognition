#!/usr/bin/env python3
"""
Combine Multiple Datasets for Training

This script combines the AT&T faces dataset with custom training data
to create a comprehensive dataset for face recognition training.

Combines:
- att_faces_dataset/ (40+ people, 10 images each)
- training_data/ (DINA + Test_Person_1, 30 images each)

Creates:
- combined_dataset/ with proper train/test split

Features:
- Automatic image resizing and normalization
- Conflict resolution strategies
- Data validation and quality checks
- Reproducible splits with seed control
"""

import argparse
import sys
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import cv2
import numpy as np

# Fix Windows terminal encoding
if sys.platform == "win32":
    try:
        import codecs
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, errors='replace')
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, errors='replace')
    except Exception:
        pass


class DatasetCombiner:
    """Handles combining multiple face recognition datasets."""
    
    def __init__(
        self,
        output_dir: str,
        test_split: float = 0.2,
        target_size: Tuple[int, int] = (100, 100),
        conflict_strategy: str = "prefer_larger"
    ):
        """
        Initialize dataset combiner.
        
        Args:
            output_dir: Output directory for combined dataset
            test_split: Fraction of data for testing (0.0-0.5)
            target_size: Target size for all images (width, height)
            conflict_strategy: How to handle conflicts ('prefer_larger', 'prefer_first', 'prefer_second')
        """
        self.output_dir = Path(output_dir)
        self.test_split = test_split
        self.target_size = target_size
        self.conflict_strategy = conflict_strategy
        
        self.stats = {
            'total_people': 0,
            'total_images': 0,
            'train_images': 0,
            'test_images': 0,
            'failed_images': 0,
            'people_with_conflicts': [],
            'skipped_people': [],
            'image_quality_issues': []
        }
        
        self.person_data = {}  # Store all person data before processing
    
    def validate_image(self, img_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Validate image quality and readability.
        
        Args:
            img_path: Path to image file
            
        Returns:
            tuple: (is_valid, error_message)
        """
        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                return False, "Could not read image"
            
            if img.size == 0:
                return False, "Empty image"
            
            # Check minimum dimensions
            if img.shape[0] < 20 or img.shape[1] < 20:
                return False, f"Image too small: {img.shape}"
            
            # Check if image is mostly blank
            if np.mean(img) < 10 or np.mean(img) > 245:
                return False, "Image appears blank or overexposed"
            
            return True, None
            
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def process_image(self, img_path: Path) -> Optional[np.ndarray]:
        """
        Read and preprocess an image.
        
        Args:
            img_path: Path to image file
            
        Returns:
            Preprocessed image or None if failed
        """
        try:
            # Read image
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            
            if img is None or img.size == 0:
                return None
            
            # Resize to target size
            img_resized = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
            
            # Apply histogram equalization for better normalization
            img_normalized = cv2.equalizeHist(img_resized)
            
            return img_normalized
            
        except Exception as e:
            print(f"      ⚠️  Error processing {img_path.name}: {e}")
            return None
    
    def collect_person_images(self, person_dir: Path, source_name: str) -> List[Path]:
        """
        Collect all valid image files from a person's directory.
        
        Args:
            person_dir: Directory containing person's images
            source_name: Name of the source dataset
            
        Returns:
            List of valid image paths
        """
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.pgm', '*.JPG', '*.JPEG', '*.PNG', '*.PGM']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(person_dir.glob(ext)))
        
        # Validate images
        valid_images = []
        for img_path in image_files:
            is_valid, error_msg = self.validate_image(img_path)
            if is_valid:
                valid_images.append(img_path)
            else:
                self.stats['failed_images'] += 1
                self.stats['image_quality_issues'].append(
                    f"{person_dir.name}/{img_path.name} ({source_name}): {error_msg}"
                )
        
        return valid_images
    
    def add_dataset(self, dataset_dir: str, dataset_name: str):
        """
        Add a dataset to the combination pool.
        
        Args:
            dataset_dir: Path to dataset directory
            dataset_name: Name of the dataset for tracking
        """
        dataset_path = Path(dataset_dir)
        
        if not dataset_path.exists():
            print(f"⚠️  Warning: Dataset not found: {dataset_dir}")
            return
        
        print(f"\n🔍 Processing {dataset_name}...")
        person_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
        
        for person_dir in sorted(person_dirs):
            person_name = person_dir.name
            
            # Collect images
            image_files = self.collect_person_images(person_dir, dataset_name)
            
            if not image_files:
                print(f"   ⚠️  No valid images for {person_name}")
                self.stats['skipped_people'].append(f"{person_name} ({dataset_name})")
                continue
            
            # Check if this is an Omar variant and merge with main OMAR
            if self._is_omar_variant(person_name):
                self._merge_with_omar(image_files, dataset_name, person_name)
            else:
                # Store or handle conflict
                if person_name in self.person_data:
                    self._handle_conflict(person_name, image_files, dataset_name)
                else:
                    self.person_data[person_name] = {
                        'images': image_files,
                        'source': dataset_name
                    }
                    print(f"   ✓ {person_name}: {len(image_files)} images from {dataset_name}")
    
    def _handle_conflict(self, person_name: str, new_images: List[Path], new_source: str):
        """Handle conflicts when a person exists in multiple datasets."""
        existing_data = self.person_data[person_name]
        existing_count = len(existing_data['images'])
        new_count = len(new_images)
        
        self.stats['people_with_conflicts'].append(person_name)
        
        if self.conflict_strategy == "prefer_larger":
            if new_count > existing_count:
                print(f"   ⚠️  {person_name}: Using {new_source} ({new_count} images) "
                      f"over {existing_data['source']} ({existing_count} images)")
                self.person_data[person_name] = {
                    'images': new_images,
                    'source': new_source
                }
            else:
                print(f"   ⚠️  {person_name}: Keeping {existing_data['source']} ({existing_count} images) "
                      f"over {new_source} ({new_count} images)")
        
        elif self.conflict_strategy == "prefer_first":
            print(f"   ⚠️  {person_name}: Keeping first source {existing_data['source']} "
                  f"({existing_count} images), skipping {new_source} ({new_count} images)")
        
        elif self.conflict_strategy == "prefer_second":
            print(f"   ⚠️  {person_name}: Using newer source {new_source} ({new_count} images) "
                  f"over {existing_data['source']} ({existing_count} images)")
            self.person_data[person_name] = {
                'images': new_images,
                'source': new_source
            }
        
        elif self.conflict_strategy == "merge":
            # Merge images from both sources
            all_images = existing_data['images'] + new_images
            print(f"   ⚠️  {person_name}: Merging {existing_data['source']} + {new_source} "
                  f"({len(all_images)} total images)")
            self.person_data[person_name] = {
                'images': all_images,
                'source': f"{existing_data['source']}+{new_source}"
            }
    
    def _is_omar_variant(self, person_name: str) -> bool:
        """
        Check if a person name is an Omar variant that should be merged.
        
        Args:
            person_name: Name of the person to check
            
        Returns:
            True if this should be merged with OMAR
        """
        omar_variants = [
            'OMAR', 'OMAR_FINAL', 'OMAR_IMPROVED', 'OMAR_ULTRA', 
            'omar', 'omar_final', 'omar_improved', 'omar_ultra'
        ]
        return person_name in omar_variants
    
    def _merge_with_omar(self, image_files: List[Path], dataset_name: str, variant_name: str):
        """
        Merge Omar variant images with the main OMAR person.
        
        Args:
            image_files: List of image files from the variant
            dataset_name: Name of the source dataset
            variant_name: Name of the Omar variant
        """
        if 'OMAR' not in self.person_data:
            # Create main OMAR entry
            self.person_data['OMAR'] = {
                'images': image_files,
                'source': f"{dataset_name} ({variant_name})"
            }
            print(f"   ✓ OMAR: {len(image_files)} images from {variant_name} ({dataset_name})")
        else:
            # Merge with existing OMAR
            existing_images = self.person_data['OMAR']['images']
            all_images = existing_images + image_files
            self.person_data['OMAR'] = {
                'images': all_images,
                'source': f"{self.person_data['OMAR']['source']} + {dataset_name} ({variant_name})"
            }
            print(f"   🔄 OMAR: Merged {len(image_files)} images from {variant_name} "
                  f"({dataset_name}) → Total: {len(all_images)} images")
    
    def create_split(self, min_train_samples: int = 1, min_test_samples: int = 1):
        """
        Create train/test split and save images.
        
        Args:
            min_train_samples: Minimum training samples per person
            min_test_samples: Minimum test samples per person (0 to disable)
        """
        print("\n" + "="*70)
        print("📂 Creating train/test split...")
        print("="*70)
        
        # Create output directories
        train_dir = self.output_dir / "train"
        test_dir = self.output_dir / "test"
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        for person_name, data in sorted(self.person_data.items()):
            images = data['images']
            source = data['source']
            
            # Shuffle for random split
            random.shuffle(images)
            
            # Calculate split ensuring minimum samples
            total = len(images)
            
            if total < min_train_samples + min_test_samples:
                print(f"⚠️  {person_name}: Only {total} images "
                      f"(need {min_train_samples + min_test_samples}), using all for training")
                train_files = images
                test_files = []
            else:
                split_idx = max(min_train_samples, int(total * (1 - self.test_split)))
                train_files = images[:split_idx]
                test_files = images[split_idx:]
                
                # Ensure minimum test samples
                if min_test_samples > 0 and len(test_files) < min_test_samples:
                    needed = min_test_samples - len(test_files)
                    test_files.extend(train_files[-needed:])
                    train_files = train_files[:-needed]
            
            # Process and save images
            train_saved = self._save_person_images(
                train_files, train_dir / person_name, person_name
            )
            test_saved = self._save_person_images(
                test_files, test_dir / person_name, person_name
            )
            
            # Update statistics
            self.stats['total_people'] += 1
            self.stats['total_images'] += len(images)
            self.stats['train_images'] += train_saved
            self.stats['test_images'] += test_saved
            
            print(f"✓ {person_name:20} Total: {total:3} → Train: {train_saved:3}, Test: {test_saved:3} ({source})")
    
    def _save_person_images(
        self, 
        image_files: List[Path], 
        output_dir: Path, 
        person_name: str
    ) -> int:
        """
        Save processed images to output directory.
        
        Args:
            image_files: List of image paths to process
            output_dir: Output directory for this person
            person_name: Person's name for naming
            
        Returns:
            Number of successfully saved images
        """
        if not image_files:
            return 0
        
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_count = 0
        
        for i, img_path in enumerate(image_files):
            img = self.process_image(img_path)
            
            if img is not None:
                # Create consistent naming
                output_name = f"{person_name}_{i:04d}.jpg"
                output_file = output_dir / output_name
                
                success = cv2.imwrite(str(output_file), img)
                if success:
                    saved_count += 1
                else:
                    print(f"      ⚠️  Failed to save: {output_name}")
                    self.stats['failed_images'] += 1
        
        return saved_count
    
    def print_statistics(self):
        """Print detailed statistics about the combined dataset."""
        print("\n" + "="*70)
        print("📊 COMBINED DATASET STATISTICS")
        print("="*70)
        print(f"Total People:        {self.stats['total_people']}")
        print(f"Total Images:        {self.stats['total_images']}")
        print(f"Training Images:     {self.stats['train_images']}")
        print(f"Test Images:         {self.stats['test_images']}")
        print(f"Failed Images:       {self.stats['failed_images']}")
        
        if self.stats['train_images'] > 0:
            train_ratio = self.stats['train_images'] / (self.stats['train_images'] + self.stats['test_images'])
            print(f"Train/Test Ratio:    {train_ratio:.1%} / {1-train_ratio:.1%}")
            print(f"Avg Images/Person:   {self.stats['total_images'] / self.stats['total_people']:.1f}")
        
        if self.stats['people_with_conflicts']:
            print(f"\n⚠️  Conflicts Resolved: {len(self.stats['people_with_conflicts'])}")
            for person in self.stats['people_with_conflicts']:
                print(f"   • {person}: Resolved using '{self.conflict_strategy}' strategy")
        
        if self.stats['skipped_people']:
            print(f"\n⚠️  Skipped People: {len(self.stats['skipped_people'])}")
            for person in self.stats['skipped_people'][:10]:  # Show first 10
                print(f"   • {person}")
            if len(self.stats['skipped_people']) > 10:
                print(f"   ... and {len(self.stats['skipped_people']) - 10} more")
        
        if self.stats['image_quality_issues']:
            print(f"\n⚠️  Image Quality Issues: {len(self.stats['image_quality_issues'])}")
            for issue in self.stats['image_quality_issues'][:5]:  # Show first 5
                print(f"   • {issue}")
            if len(self.stats['image_quality_issues']) > 5:
                print(f"   ... and {len(self.stats['image_quality_issues']) - 5} more")
        
        print("\n✅ Dataset combination complete!")
        print(f"📁 Combined dataset saved to: {self.output_dir}")


def create_training_script(output_dir: Path, model_name: str = "combined_model"):
    """Create a training script for the combined dataset."""
    
    script_content = f'''#!/usr/bin/env python3
"""
Train Model on Combined Dataset

This script trains a face recognition model using the combined dataset.

Usage:
    python train_{model_name}.py
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
        data_dir="{output_dir}/train",
        output_path="models/{model_name}.yml",
        test_split=0.0,  # No split needed, we already have separate test set
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8,
        threshold=50.0,
        use_enhanced_preprocessing=True
    )
    
    if recognizer is None:
        print("\\n❌ Training failed!")
        return 1
    
    print("\\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print("\\nNext steps:")
    print("1. Evaluate on test set:")
    print("   python evaluate_{model_name}.py")
    print("\\n2. Test with ultra-strict recognition:")
    print("   python examples/ultra_strict_recognition.py --model models/{model_name}.yml")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''
    
    script_path = output_dir / f"train_{model_name}.py"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    # Make executable on Unix-like systems
    try:
        script_path.chmod(0o755)
    except Exception:
        pass
    
    print(f"📝 Created training script: {script_path}")


def create_evaluation_script(output_dir: Path, model_name: str = "combined_model"):
    """Create an evaluation script for the combined dataset."""
    
    script_content = f'''#!/usr/bin/env python3
"""
Evaluate Combined Model on Test Set

This script evaluates the trained model on the held-out test set.

Usage:
    python evaluate_{model_name}.py
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
    label_names = {{}}
    
    person_dirs = sorted([d for d in test_dir.iterdir() if d.is_dir()])
    
    print(f"📂 Loading test data from {{len(person_dirs)}} people...\\n")
    
    for label_id, person_dir in enumerate(person_dirs):
        person_name = person_dir.name
        label_names[label_id] = person_name
        
        image_files = list(person_dir.glob("*.jpg"))
        
        for img_file in image_files:
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                test_images.append(img)
                test_labels.append(label_id)
        
        print(f"   ✓ {{person_name}}: {{len([f for f in image_files if cv2.imread(str(f)) is not None])}} test images")
    
    return test_images, test_labels, label_names


def evaluate_model(
    recognizer: FaceRecognizer,
    test_images: List[np.ndarray],
    test_labels: List[int],
    label_names: Dict[int, str]
):
    """Evaluate model on test set."""
    
    print("\\n" + "="*70)
    print("🔍 EVALUATING MODEL ON TEST SET")
    print("="*70 + "\\n")
    
    correct = 0
    total = len(test_images)
    confidences = []
    
    # Per-person metrics
    person_correct = {{label: 0 for label in set(test_labels)}}
    person_total = {{label: 0 for label in set(test_labels)}}
    
    for i, (img, true_label) in enumerate(zip(test_images, test_labels)):
        try:
            pred_label, confidence = recognizer.predict(img, return_confidence=True)
        except Exception as e:
            print(f"⚠️  Error predicting sample {{i+1}}: {{e}}")
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
        
        true_name = label_names.get(true_label, f"ID_{{true_label}}")
        pred_name = label_names.get(pred_label, "Unknown") if pred_label != -1 else "Unknown"
        status = "✓" if is_correct else "✗"
        
        if i < 20 or not is_correct:  # Show first 20 and all errors
            print(f"{{status}} Test {{i+1}}/{{total}}: {{true_name:20}} → {{pred_name:20}} (conf: {{confidence:5.1f}})")
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
    print("\\n" + "="*70)
    print("📊 EVALUATION RESULTS")
    print("="*70)
    print(f"Total Test Images:      {{total}}")
    print(f"Correct Predictions:    {{correct}} ({{accuracy:.2f}}%)")
    print(f"Incorrect Predictions:  {{total - correct}} ({{100-accuracy:.2f}}%)")
    print()
    print(f"False Acceptance Rate:  {{far:.2f}}% (goal: ≤2%)")
    print(f"False Rejection Rate:   {{frr:.2f}}% (goal: ≤5%)")
    print()
    print(f"Average Confidence:     {{np.mean(confidences):.2f}}")
    print(f"Confidence Std Dev:     {{np.std(confidences):.2f}}")
    
    # Per-person accuracy
    print("\\n" + "-"*70)
    print("PER-PERSON ACCURACY (Top 10 and Bottom 10)")
    print("-"*70)
    
    person_accuracies = {{
        label: (person_correct[label] / person_total[label] * 100) if person_total[label] > 0 else 0
        for label in person_total.keys()
    }}
    
    sorted_accuracies = sorted(person_accuracies.items(), key=lambda x: x[1], reverse=True)
    
    # Top 10
    for label, acc in sorted_accuracies[:10]:
        name = label_names.get(label, f"ID_{{label}}")
        print(f"✅ {{name:20}} {{person_correct[label]:3}}/{{person_total[label]:3}} ({{acc:5.1f}}%)")
    
    if len(sorted_accuracies) > 20:
        print("   ...")
    
    # Bottom 10
    for label, acc in sorted_accuracies[-10:]:
        name = label_names.get(label, f"ID_{{label}}")
        print(f"⚠️  {{name:20}} {{person_correct[label]:3}}/{{person_total[label]:3}} ({{acc:5.1f}}%)")
    
    print("="*70)
    
    # Final verdict
    if accuracy >= 90 and far <= 2 and frr <= 5:
        print("\\n✅ MODEL PASSED ALL EVALUATION CRITERIA!")
        print("   Ready for deployment!")
    elif accuracy >= 90:
        print("\\n✅ MODEL PASSED ACCURACY TARGET!")
        print("   ⚠️  But check FAR/FRR values")
    else:
        print("\\n⚠️  MODEL NEEDS IMPROVEMENT")
        print("   Consider capturing more/better training data")
    
    print("="*70)
    
    return accuracy >= 90


def main():
    print("="*70)
    print("📊 EVALUATING COMBINED MODEL ON TEST SET")
    print("="*70)
    
    # Load model
    model_path = Path("models/{model_name}.yml")
    if not model_path.exists():
        print(f"\\n❌ Model not found: {{model_path}}")
        print("   Run train_{model_name}.py first!")
        return 1
    
    recognizer = FaceRecognizer(threshold=50.0, use_enhanced_preprocessing=True)
    result = recognizer.load_model(str(model_path))
    
    if not result.get('model_loaded'):
        print("\\n❌ Failed to load model!")
        return 1
    
    print(f"\\n✅ Model loaded successfully")
    print(f"📋 Known people: {{len(result.get('label_mappings', {{}}))}}")
    
    # Load test data
    test_dir = Path("{output_dir}/test")
    if not test_dir.exists():
        print(f"\\n❌ Test directory not found: {{test_dir}}")
        return 1
    
    test_images, test_labels, label_names = load_test_data(test_dir)
    
    if not test_images:
        print("\\n❌ No test images found!")
        return 1
    
    # Evaluate
    passed = evaluate_model(recognizer, test_images, test_labels, label_names)
    
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
'''
    
    script_path = output_dir / f"evaluate_{model_name}.py"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    # Make executable on Unix-like systems
    try:
        script_path.chmod(0o755)
    except Exception:
        pass
    
    print(f"📝 Created evaluation script: {script_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine multiple face recognition datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--att-dataset",
        type=str,
        default="att_faces_dataset",
        help="Path to AT&T faces dataset"
    )
    parser.add_argument(
        "--training-data",
        type=str,
        default="training_data",
        help="Path to custom training data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="combined_dataset",
        help="Output directory for combined dataset"
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Fraction of data for testing (0.0-0.5)"
    )
    parser.add_argument(
        "--target-size",
        type=int,
        nargs=2,
        default=[100, 100],
        metavar=("WIDTH", "HEIGHT"),
        help="Target size for images"
    )
    parser.add_argument(
        "--conflict-strategy",
        type=str,
        choices=["prefer_larger", "prefer_first", "prefer_second", "merge"],
        default="prefer_larger",
        help="Strategy for handling conflicts"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="combined_model",
        help="Name for generated model and scripts"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits"
    )
    parser.add_argument(
        "--min-train-samples",
        type=int,
        default=1,
        help="Minimum training samples per person"
    )
    parser.add_argument(
        "--min-test-samples",
        type=int,
        default=1,
        help="Minimum test samples per person (0 to disable)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not 0.0 <= args.test_split <= 0.5:
        parser.error("--test-split must be between 0.0 and 0.5")
    
    if args.target_size[0] < 20 or args.target_size[1] < 20:
        parser.error("--target-size dimensions must be at least 20x20")
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print("="*70)
    print("🔄 COMBINING DATASETS FOR TRAINING")
    print("="*70)
    print(f"📂 AT&T Dataset:        {args.att_dataset}")
    print(f"📂 Training Data:       {args.training_data}")
    print(f"📁 Output:              {args.output}")
    print(f"📊 Test Split:          {args.test_split:.1%}")
    print(f"🎯 Target Size:         {args.target_size[0]}x{args.target_size[1]}")
    print(f"⚔️  Conflict Strategy:   {args.conflict_strategy}")
    print(f"🎲 Random Seed:         {args.seed}")
    print("="*70)
    
    # Create combiner
    combiner = DatasetCombiner(
        output_dir=args.output,
        test_split=args.test_split,
        target_size=tuple(args.target_size),
        conflict_strategy=args.conflict_strategy
    )
    
    # Add datasets
    combiner.add_dataset(args.att_dataset, "AT&T Dataset")
    combiner.add_dataset(args.training_data, "Custom Training Data")
    
    # Check if we have any data
    if not combiner.person_data:
        print("\n❌ Error: No valid person data found in any dataset!")
        return 1
    
    # Create split and save
    combiner.create_split(
        min_train_samples=args.min_train_samples,
        min_test_samples=args.min_test_samples
    )
    
    # Print statistics
    combiner.print_statistics()
    
    # Create helper scripts
    output_path = Path(args.output)
    create_training_script(output_path, args.model_name)
    create_evaluation_script(output_path, args.model_name)
    
    # Print next steps
    print("\n" + "="*70)
    print("🚀 READY TO TRAIN!")
    print("="*70)
    print(f"\n1. Train the model:")
    print(f"   python {args.output}/train_{args.model_name}.py")
    print(f"\n2. Evaluate on test set:")
    print(f"   python {args.output}/evaluate_{args.model_name}.py")
    print(f"\n3. Test with live recognition:")
    print(f"   python examples/ultra_strict_recognition.py --model models/{args.model_name}.yml")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())