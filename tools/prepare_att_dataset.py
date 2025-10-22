#!/usr/bin/env python3
"""
Prepare AT&T Faces Dataset for Testing

Extracts and prepares the AT&T Database of Faces (ORL) for testing
the face recognition system with configurable train-test split.

Dataset: 40 people, 10 images each (400 total)
Perfect for validating the system works!

Usage:
    python tools/prepare_att_dataset.py --input att_faces.tar.Z --output att_faces_dataset
    python tools/prepare_att_dataset.py --input att_faces/ --output att_faces_dataset --already-extracted
    
Note: .tar.Z files use Unix compress format. The script will try:
1. Using unlzw library (pip install unlzw)
2. Using 7-Zip if installed
3. Using WinRAR if installed
4. Manual extraction instructions
"""

import argparse
import tarfile
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple, List
import cv2
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class ATTDatasetPreparer:
    """Handles extraction and preparation of AT&T faces dataset."""
    
    def __init__(
        self,
        output_dir: str,
        target_size: Tuple[int, int] = (100, 100),
        normalize: bool = True
    ):
        """
        Initialize dataset preparer.
        
        Args:
            output_dir: Output directory for processed dataset
            target_size: Target size for all images (width, height)
            normalize: Whether to apply histogram equalization
        """
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self.normalize = normalize
        
        self.stats = {
            'total_people': 0,
            'total_images': 0,
            'failed_images': 0,
            'person_details': []
        }
    
    def decompress_tar_z(self, tar_z_path: Path, output_tar_path: Path) -> bool:
        """
        Decompress .tar.Z file to .tar
        
        .tar.Z uses Unix compress (LZW), not supported by Python tarfile directly.
        Tries multiple methods.
        
        Args:
            tar_z_path: Path to .tar.Z file
            output_tar_path: Path for output .tar file
            
        Returns:
            bool: True if successful, False otherwise
        """
        print("🔧 Decompressing .tar.Z file...")
        
        # Method 1: Try unlzw library (Python 3 compatible)
        try:
            import unlzw  # pyright: ignore[reportMissingImports]
            print("   • Trying unlzw library...")
            
            with open(tar_z_path, 'rb') as f_in:
                compressed_data = f_in.read()
            
            decompressed_data = unlzw.unlzw(compressed_data)
            
            with open(output_tar_path, 'wb') as f_out:
                f_out.write(decompressed_data)
            
            print("   ✅ Decompressed using unlzw")
            return True
        except ImportError:
            print("   ⚠️  unlzw not installed (pip install unlzw)")
        except Exception as e:
            print(f"   ⚠️  unlzw failed: {e}")
        
        # Method 2: Try system uncompress command (Unix/Linux/Mac)
        if sys.platform != "win32":
            try:
                print("   • Trying system 'uncompress' command...")
                
                # Copy to temp location for uncompress
                temp_z = output_tar_path.parent / tar_z_path.name
                shutil.copy2(tar_z_path, temp_z)
                
                result = subprocess.run(
                    ["uncompress", str(temp_z)],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    # uncompress removes .Z and creates .tar
                    uncompressed = temp_z.with_suffix('')
                    if uncompressed.exists():
                        shutil.move(str(uncompressed), str(output_tar_path))
                        print("   ✅ Decompressed using system uncompress")
                        return True
            except FileNotFoundError:
                print("   ⚠️  'uncompress' command not found")
            except Exception as e:
                print(f"   ⚠️  System uncompress failed: {e}")
        
        # Method 3: Try 7-Zip (cross-platform)
        try:
            print("   • Trying 7-Zip...")
            
            seven_zip_paths = [
                "7z",  # In PATH
                "7za",  # Alternative command
                r"C:\Program Files\7-Zip\7z.exe",
                r"C:\Program Files (x86)\7-Zip\7z.exe",
                "/usr/bin/7z",
                "/usr/local/bin/7z"
            ]
            
            seven_zip_exe = None
            for path in seven_zip_paths:
                try:
                    # Test if command works
                    result = subprocess.run(
                        [path, "--help"],
                        capture_output=True,
                        timeout=2
                    )
                    if result.returncode == 0:
                        seven_zip_exe = path
                        break
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    continue
            
            if seven_zip_exe:
                # Extract to temp location
                temp_dir = output_tar_path.parent / "temp_7z"
                temp_dir.mkdir(exist_ok=True)
                
                result = subprocess.run(
                    [seven_zip_exe, "e", str(tar_z_path), f"-o{temp_dir}", "-y"],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    # Find the extracted .tar file
                    tar_files = list(temp_dir.glob("*.tar"))
                    if tar_files:
                        shutil.move(str(tar_files[0]), str(output_tar_path))
                        shutil.rmtree(temp_dir)
                        print("   ✅ Decompressed using 7-Zip")
                        return True
                
                shutil.rmtree(temp_dir, ignore_errors=True)
            else:
                print("   ⚠️  7-Zip not found")
        except Exception as e:
            print(f"   ⚠️  7-Zip failed: {e}")
        
        # Method 4: Try WinRAR (Windows only)
        if sys.platform == "win32":
            try:
                print("   • Trying WinRAR...")
                
                winrar_paths = [
                    r"C:\Program Files\WinRAR\WinRAR.exe",
                    r"C:\Program Files (x86)\WinRAR\WinRAR.exe",
                    r"C:\Program Files\WinRAR\UnRAR.exe",
                    r"C:\Program Files (x86)\WinRAR\UnRAR.exe"
                ]
                
                winrar_exe = None
                for path in winrar_paths:
                    if Path(path).exists():
                        winrar_exe = path
                        break
                
                if winrar_exe:
                    temp_dir = output_tar_path.parent / "temp_winrar"
                    temp_dir.mkdir(exist_ok=True)
                    
                    result = subprocess.run(
                        [winrar_exe, "x", "-y", str(tar_z_path), str(temp_dir)],
                        capture_output=True,
                        text=True
                    )
                    
                    if result.returncode == 0:
                        tar_files = list(temp_dir.glob("*.tar"))
                        if tar_files:
                            shutil.move(str(tar_files[0]), str(output_tar_path))
                            shutil.rmtree(temp_dir)
                            print("   ✅ Decompressed using WinRAR")
                            return True
                    
                    shutil.rmtree(temp_dir, ignore_errors=True)
                else:
                    print("   ⚠️  WinRAR not found")
            except Exception as e:
                print(f"   ⚠️  WinRAR failed: {e}")
        
        return False
    
    def print_manual_instructions(self, tar_z_path: Path):
        """Print manual extraction instructions."""
        print("\n" + "="*70)
        print("❌ AUTOMATIC DECOMPRESSION FAILED")
        print("="*70)
        print("\n📋 MANUAL EXTRACTION INSTRUCTIONS:")
        
        print("\n🐍 Option 1: Install unlzw library (Recommended)")
        print("  pip install unlzw")
        print("  Then run this script again")
        
        print("\n🗜️  Option 2: Use 7-Zip (Best for Windows)")
        print("  1. Download: https://www.7-zip.org/")
        print("  2. Install 7-Zip")
        print(f"  3. Right-click {tar_z_path.name}")
        print("  4. Select '7-Zip' → 'Extract Here'")
        print("  5. Extract the resulting .tar file again")
        print("  6. Run this script with --already-extracted:")
        print(f"     python {Path(__file__).name} --input <extracted_folder> --output {self.output_dir} --already-extracted")
        
        if sys.platform != "win32":
            print("\n🐧 Option 3: Use system uncompress (Linux/Mac)")
            print(f"  uncompress {tar_z_path}")
            print(f"  tar -xf {tar_z_path.with_suffix('')}")
            print(f"  python {Path(__file__).name} --input <extracted_folder> --output {self.output_dir} --already-extracted")
        
        print("\n🌐 Option 4: Use online converter")
        print("  1. Upload to: https://extract.me/ or https://www.ezyzip.com/")
        print("  2. Download extracted files")
        print("  3. Run script with --already-extracted")
        
        print("\n📥 Option 5: Download pre-extracted dataset")
        print("  The dataset is also available pre-extracted from:")
        print("  https://www.kaggle.com/datasets/kasikrit/att-database-of-faces")
        print("  Download and use --already-extracted flag")
        
        print("="*70)
    
    def validate_image(self, img_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Validate image quality.
        
        Args:
            img_path: Path to image
            
        Returns:
            tuple: (is_valid, error_message)
        """
        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                return False, "Could not read image"
            
            if img.size == 0:
                return False, "Empty image"
            
            # Check minimum size
            if img.shape[0] < 20 or img.shape[1] < 20:
                return False, f"Too small: {img.shape}"
            
            # Check if mostly blank
            mean_val = np.mean(img)
            if mean_val < 5 or mean_val > 250:
                return False, f"Unusual brightness: {mean_val:.1f}"
            
            return True, None
            
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def process_image(self, img_path: Path) -> Optional[np.ndarray]:
        """
        Process a single image.
        
        Args:
            img_path: Path to image
            
        Returns:
            Processed image or None if failed
        """
        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            
            if img is None or img.size == 0:
                return None
            
            # Resize to target size
            img_resized = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
            
            # Apply normalization if enabled
            if self.normalize:
                img_resized = cv2.equalizeHist(img_resized)
            
            return img_resized
            
        except Exception as e:
            print(f"      ⚠️  Error processing {img_path.name}: {e}")
            return None
    
    def process_person_directory(self, person_dir: Path, person_name: str) -> int:
        """
        Process all images for a person.
        
        Args:
            person_dir: Directory containing person's images
            person_name: Name for the person
            
        Returns:
            Number of successfully processed images
        """
        # Create output directory
        person_output = self.output_dir / person_name
        person_output.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_extensions = ['*.pgm', '*.jpg', '*.jpeg', '*.png', '*.PGM', '*.JPG', '*.JPEG', '*.PNG']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(person_dir.glob(ext)))
        
        if not image_files:
            print(f"  ⚠️  No images found for {person_name}")
            return 0
        
        # Process each image
        processed = 0
        for img_file in sorted(image_files):
            # Validate
            is_valid, error = self.validate_image(img_file)
            if not is_valid:
                print(f"  ⚠️  {img_file.name}: {error}")
                self.stats['failed_images'] += 1
                continue
            
            # Process
            img = self.process_image(img_file)
            if img is None:
                self.stats['failed_images'] += 1
                continue
            
            # Save with consistent naming
            output_name = f"{person_name}_{img_file.stem}.jpg"
            output_file = person_output / output_name
            
            success = cv2.imwrite(str(output_file), img)
            if success:
                processed += 1
            else:
                print(f"  ⚠️  Failed to save: {output_name}")
                self.stats['failed_images'] += 1
        
        return processed
    
    def extract_and_process(
        self,
        input_path: str,
        already_extracted: bool = False
    ) -> bool:
        """
        Main extraction and processing pipeline.
        
        Args:
            input_path: Path to input file or directory
            already_extracted: Whether input is already extracted
            
        Returns:
            bool: True if successful
        """
        input_path_obj = Path(input_path)
        
        if not input_path_obj.exists():
            print(f"❌ Error: Input not found: {input_path}")
            return False
        
        print("="*70)
        print("AT&T FACES DATASET PREPARATION")
        print("="*70 + "\n")
        print(f"📦 Input:  {input_path_obj}")
        print(f"📁 Output: {self.output_dir}")
        print(f"🎯 Target Size: {self.target_size[0]}x{self.target_size[1]}")
        print(f"🔧 Normalization: {'Enabled' if self.normalize else 'Disabled'}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Handle different input types
        dataset_dir = None
        
        if already_extracted and input_path_obj.is_dir():
            print("\n✅ Using already extracted dataset")
            dataset_dir = input_path_obj
            
        elif input_path_obj.is_dir():
            print("\n📂 Processing directory...")
            dataset_dir = input_path_obj
            
        elif input_path_obj.suffix == '.Z':
            print(f"\n📝 Detected .tar.Z format (Unix compress)")
            
            # Try to decompress
            temp_tar = self.output_dir / "temp_att_faces.tar"
            
            if not self.decompress_tar_z(input_path_obj, temp_tar):
                self.print_manual_instructions(input_path_obj)
                return False
            
            # Extract tar
            print(f"\n📦 Extracting tar archive...")
            try:
                with tarfile.open(temp_tar, 'r:*') as tar:
                    # Extract to temp directory
                    temp_extract = self.output_dir / "temp_extract"
                    temp_extract.mkdir(exist_ok=True)
                    tar.extractall(temp_extract)
                
                print("✅ Extraction complete!")
                
                # Find the dataset directory
                extracted_dirs = [d for d in temp_extract.iterdir() if d.is_dir()]
                if extracted_dirs:
                    dataset_dir = extracted_dirs[0]
                else:
                    dataset_dir = temp_extract
                
                # Clean up temp tar
                temp_tar.unlink(missing_ok=True)
                
            except Exception as e:
                print(f"❌ Extraction failed: {e}")
                temp_tar.unlink(missing_ok=True)
                return False
                
        elif input_path_obj.suffix in ['.tar', '.gz', '.tgz']:
            print(f"\n📦 Extracting {input_path_obj.suffix} archive...")
            try:
                with tarfile.open(input_path_obj, 'r:*') as tar:
                    temp_extract = self.output_dir / "temp_extract"
                    temp_extract.mkdir(exist_ok=True)
                    tar.extractall(temp_extract)
                
                print("✅ Extraction complete!")
                
                extracted_dirs = [d for d in temp_extract.iterdir() if d.is_dir()]
                if extracted_dirs:
                    dataset_dir = extracted_dirs[0]
                else:
                    dataset_dir = temp_extract
                    
            except Exception as e:
                print(f"❌ Extraction failed: {e}")
                return False
        else:
            print(f"❌ Unsupported file format: {input_path_obj.suffix}")
            return False
        
        if dataset_dir is None:
            print("❌ Could not find dataset directory")
            return False
        
        print(f"\n📂 Processing from: {dataset_dir.name}")
        
        # Find person directories
        # AT&T dataset typically has directories named s1, s2, ..., s40
        person_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
        
        if not person_dirs:
            print("❌ No person directories found")
            print(f"   Searched in: {dataset_dir}")
            print(f"   Contents: {list(dataset_dir.iterdir())[:10]}")
            return False
        
        print(f"\n👥 Found {len(person_dirs)} people in dataset")
        print("🔄 Processing images...\n")
        
        # Process each person
        for person_dir in person_dirs:
            # Generate person name (convert s1 → Person_01, etc.)
            if person_dir.name.startswith('s') and person_dir.name[1:].isdigit():
                person_num = int(person_dir.name[1:])
                person_name = f"Person_{person_num:02d}"
            else:
                # Fallback to directory name
                person_name = person_dir.name.replace('_', ' ').replace('-', ' ').title().replace(' ', '_')
            
            # Process images
            processed = self.process_person_directory(person_dir, person_name)
            
            if processed > 0:
                self.stats['total_people'] += 1
                self.stats['total_images'] += processed
                self.stats['person_details'].append({
                    'name': person_name,
                    'images': processed
                })
                print(f"  ✓ {person_name}: {processed} images processed")
            else:
                print(f"  ✗ {person_name}: No images processed")
        
        # Clean up temporary extraction
        if 'temp_extract' in str(dataset_dir):
            print(f"\n🧹 Cleaning up temporary files...")
            shutil.rmtree(dataset_dir.parent / "temp_extract", ignore_errors=True)
        
        return True
    
    def print_statistics(self):
        """Print processing statistics."""
        print(f"\n" + "="*70)
        print("📊 PROCESSING STATISTICS")
        print("="*70)
        print(f"Total People:     {self.stats['total_people']}")
        print(f"Total Images:     {self.stats['total_images']}")
        print(f"Failed Images:    {self.stats['failed_images']}")
        
        if self.stats['total_people'] > 0:
            avg_images = self.stats['total_images'] / self.stats['total_people']
            print(f"Avg Images/Person: {avg_images:.1f}")
        
        print(f"\nOutput Directory: {self.output_dir}")
        
        if self.stats['failed_images'] > 0:
            print(f"\n⚠️  {self.stats['failed_images']} images failed to process")
        
        print("="*70)
    
    def print_next_steps(self, model_name: str = "att_test_model"):
        """Print next steps for using the dataset."""
        print("\n" + "="*70)
        print("🚀 NEXT STEPS")
        print("="*70)
        
        total = self.stats['total_images']
        train_split = int(total * 0.8)
        test_split = total - train_split
        
        print(f"\n1️⃣  Train with 80/20 split:")
        print(f"   python examples/train_with_validation.py \\")
        print(f"     --data {self.output_dir} \\")
        print(f"     --output models/{model_name}.yml \\")
        print(f"     --split 0.2")
        
        print(f"\n   Expected split:")
        print(f"   • Training: ~{train_split} images")
        print(f"   • Testing:  ~{test_split} images")
        print(f"   • Expected accuracy: 90-95%")
        
        print(f"\n2️⃣  Test live recognition:")
        print(f"   python examples/live_recognition.py \\")
        print(f"     --model models/{model_name}.yml")
        
        print(f"\n3️⃣  Combine with custom data:")
        print(f"   python tools/combine_datasets.py \\")
        print(f"     --att-dataset {self.output_dir} \\")
        print(f"     --training-data your_training_data \\")
        print(f"     --output combined_dataset")
        
        print(f"\n⚠️  IMPORTANT NOTES:")
        print(f"   • This dataset is for TESTING and VALIDATION only")
        print(f"   • For actual attendance, capture YOUR people's faces")
        print(f"   • AT&T dataset: {self.stats['total_people']} people, ~{total} images total")
        print(f"   • Images are grayscale, {self.target_size[0]}x{self.target_size[1]} pixels")
        
        print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare AT&T Faces Dataset for face recognition training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to att_faces.tar.Z file or extracted directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="att_faces_dataset",
        help="Output directory for processed dataset"
    )
    parser.add_argument(
        "--already-extracted",
        action="store_true",
        help="Input is already extracted (skip decompression)"
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
        "--no-normalize",
        action="store_true",
        help="Disable histogram equalization"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="att_test_model",
        help="Name for the model (used in instructions)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.target_size[0] < 20 or args.target_size[1] < 20:
        parser.error("--target-size dimensions must be at least 20x20")
    
    # Create preparer
    preparer = ATTDatasetPreparer(
        output_dir=args.output,
        target_size=tuple(args.target_size),
        normalize=not args.no_normalize
    )
    
    # Process dataset
    success = preparer.extract_and_process(
        input_path=args.input,
        already_extracted=args.already_extracted
    )
    
    if success:
        preparer.print_statistics()
        preparer.print_next_steps(args.model_name)
        print("\n✅ Dataset preparation complete!")
        return 0
    else:
        print("\n❌ Dataset preparation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())