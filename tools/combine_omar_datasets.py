#!/usr/bin/env python3
"""
Combine All Omar Datasets into a Single Person

This script combines all Omar variants (OMAR, OMAR_FINAL, OMAR_IMPROVED, OMAR_ULTRA)
from different datasets into a single OMAR person for training.

Usage:
    python tools/combine_omar_datasets.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.combine_datasets import DatasetCombiner


def main():
    print("="*70)
    print("🔄 COMBINING ALL OMAR DATASETS INTO SINGLE PERSON")
    print("="*70)
    
    # Create combiner with merge strategy for Omar variants
    combiner = DatasetCombiner(
        output_dir="unified_omar_dataset",
        test_split=0.2,
        target_size=(100, 100),
        conflict_strategy="merge"  # Use merge strategy to combine all Omar variants
    )
    
    # Add all datasets that contain Omar variants
    datasets_to_combine = [
        ("final_omar_dataset", "Final Omar Dataset"),
        ("clean_omar_dataset", "Clean Omar Dataset"), 
        ("omar_dataset", "Omar Dataset"),
        ("training_data", "Training Data"),
        ("att_faces_dataset", "AT&T Dataset")
    ]
    
    print(f"📂 Combining datasets:")
    for dataset_path, dataset_name in datasets_to_combine:
        print(f"   • {dataset_name}: {dataset_path}")
    
    print("="*70)
    
    # Add each dataset
    for dataset_path, dataset_name in datasets_to_combine:
        combiner.add_dataset(dataset_path, dataset_name)
    
    # Check if we have any data
    if not combiner.person_data:
        print("\n❌ Error: No valid person data found in any dataset!")
        return 1
    
    # Create split and save
    combiner.create_split(
        min_train_samples=5,  # Ensure good training samples
        min_test_samples=2    # Ensure test samples
    )
    
    # Print statistics
    combiner.print_statistics()
    
    # Print Omar-specific statistics
    if 'OMAR' in combiner.person_data:
        omar_data = combiner.person_data['OMAR']
        print("\n" + "="*70)
        print("👤 OMAR COMBINATION SUMMARY")
        print("="*70)
        print(f"Total Omar Images:     {len(omar_data['images'])}")
        print(f"Source Datasets:       {omar_data['source']}")
        print(f"Train/Test Split:      {len(omar_data['images']) * 0.8:.0f} train, {len(omar_data['images']) * 0.2:.0f} test")
        print("="*70)
    
    # Create helper scripts
    output_path = Path("unified_omar_dataset")
    from tools.combine_datasets import create_training_script, create_evaluation_script
    create_training_script(output_path, "unified_omar_model")
    create_evaluation_script(output_path, "unified_omar_model")
    
    # Print next steps
    print("\n" + "="*70)
    print("🚀 READY TO TRAIN UNIFIED OMAR MODEL!")
    print("="*70)
    print(f"\n1. Train the unified model:")
    print(f"   python unified_omar_dataset/train_unified_omar_model.py")
    print(f"\n2. Evaluate on test set:")
    print(f"   python unified_omar_dataset/evaluate_unified_omar_model.py")
    print(f"\n3. Test with live recognition:")
    print(f"   python examples/ultra_strict_recognition.py --model models/unified_omar_model.yml")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
