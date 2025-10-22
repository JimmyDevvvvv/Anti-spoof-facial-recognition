# Face Recognition API Documentation

## Overview

The Face Recognition API implements the Local Binary Pattern Histogram (LBPH) algorithm for robust facial recognition. This module provides comprehensive functionality for training recognition models, predicting identities, and managing face encodings with high accuracy and efficiency.

## Table of Contents
1. [FaceRecognizer Class](#facerecognizer-class)
2. [LBPH Algorithm Details](#lbph-algorithm-details)
3. [Training and Model Management](#training-and-model-management)
4. [Recognition and Prediction](#recognition-and-prediction)
5. [Performance Optimization](#performance-optimization)
6. [Advanced Features](#advanced-features)
7. [Integration Examples](#integration-examples)

---

## FaceRecognizer Class

### Class Definition

```python
class FaceRecognizer:
    """
    Advanced face recognition system using Local Binary Pattern Histogram (LBPH) algorithm.
    
    This class provides comprehensive face recognition capabilities including training,
    prediction, model management, and performance optimization features.
    """
    
    def __init__(self, radius=1, neighbors=8, grid_x=8, grid_y=8, threshold=100.0,
                 enable_preprocessing=True, enable_confidence_calibration=True):
        """
        Initialize the LBPH Face Recognizer with specified parameters.
        
        Args:
            radius (int): Radius for LBP calculation (typically 1-3)
            neighbors (int): Number of sample points for LBP (typically 8, 16, 24)
            grid_x (int): Number of cells in horizontal direction (4-16)
            grid_y (int): Number of cells in vertical direction (4-16)
            threshold (float): Recognition confidence threshold (lower = more strict)
            enable_preprocessing (bool): Enable automatic image preprocessing
            enable_confidence_calibration (bool): Enable confidence score calibration
        
        Example:
            >>> # Standard configuration
            >>> recognizer = FaceRecognizer()
            >>> 
            >>> # High accuracy configuration
            >>> recognizer = FaceRecognizer(
            ...     radius=2,
            ...     neighbors=16,
            ...     grid_x=10,
            ...     grid_y=10,
            ...     threshold=80.0
            ... )
        """
```

### Core Methods

#### train()

```python
def train(self, face_samples, labels, validation_split=0.2, 
         augment_data=False, save_backup=True):
    """
    Train the face recognition model with provided samples.
    
    Args:
        face_samples (list): List of face images (grayscale numpy arrays)
        labels (list): Corresponding integer labels/IDs for each face sample
        validation_split (float): Fraction of data to use for validation (0.0-0.5)
        augment_data (bool): Whether to apply data augmentation techniques
        save_backup (bool): Whether to save a backup of previous model
    
    Returns:
        dict: Training results containing:
            - training_accuracy (float): Accuracy on training set
            - validation_accuracy (float): Accuracy on validation set
            - training_time (float): Time taken for training in seconds
            - samples_processed (int): Number of samples used for training
            - model_size (int): Size of trained model in bytes
    
    Raises:
        ValueError: If face_samples and labels have different lengths
        TrainingError: If training fails due to insufficient data
        
    Example:
        >>> # Prepare training data
        >>> face_samples = []
        >>> labels = []
        >>> 
        >>> # Load face samples for each person
        >>> for person_id in range(1, 11):  # 10 people
        ...     for sample_idx in range(20):  # 20 samples each
        ...         face_img = load_face_sample(person_id, sample_idx)
        ...         face_samples.append(face_img)
        ...         labels.append(person_id)
        >>> 
        >>> # Train the model
        >>> results = recognizer.train(
        ...     face_samples, 
        ...     labels,
        ...     validation_split=0.2,
        ...     augment_data=True
        ... )
        >>> 
        >>> print(f"Training accuracy: {results['training_accuracy']:.2f}")
        >>> print(f"Validation accuracy: {results['validation_accuracy']:.2f}")
    """
```

#### predict()

```python
def predict(self, face_image, return_confidence=True, return_all_matches=False,
           confidence_threshold=None):
    """
    Predict the identity of a face in the given image.
    
    Args:
        face_image (numpy.ndarray): Grayscale face image
        return_confidence (bool): Whether to return confidence score
        return_all_matches (bool): Whether to return all possible matches
        confidence_threshold (float, optional): Override default threshold
    
    Returns:
        tuple or dict: Depending on parameters:
            - If return_all_matches=False: (label, confidence) or label only
            - If return_all_matches=True: dict with all match results
    
    Example:
        >>> # Basic prediction
        >>> label, confidence = recognizer.predict(face_region)
        >>> 
        >>> if confidence < recognizer.threshold:
        ...     print(f"Recognized person ID: {label} (confidence: {confidence:.2f})")
        ... else:
        ...     print("Unknown person")
        >>> 
        >>> # Get all possible matches
        >>> results = recognizer.predict(face_region, return_all_matches=True)
        >>> print("Top 3 matches:")
        >>> for match in results['all_matches'][:3]:
        ...     print(f"  ID {match['label']}: {match['confidence']:.2f}")
    """
```

#### predict_batch()

```python
def predict_batch(self, face_images, batch_size=32, return_confidences=True,
                 progress_callback=None):
    """
    Perform batch prediction on multiple face images for improved efficiency.
    
    Args:
        face_images (list): List of grayscale face images
        batch_size (int): Number of images to process in each batch
        return_confidences (bool): Whether to return confidence scores
        progress_callback (function, optional): Callback for progress updates
    
    Returns:
        list: Prediction results for all input images
        
    Example:
        >>> # Process multiple faces efficiently
        >>> face_batch = [face1, face2, face3, ...]
        >>> 
        >>> def progress_update(current, total):
        ...     print(f"Progress: {current}/{total} ({100*current/total:.1f}%)")
        >>> 
        >>> predictions = recognizer.predict_batch(
        ...     face_batch,
        ...     batch_size=16,
        ...     progress_callback=progress_update
        ... )
        >>> 
        >>> for i, (label, confidence) in enumerate(predictions):
        ...     print(f"Face {i}: ID {label} (confidence: {confidence:.2f})")
    """
```

### Model Management

#### save_model()

```python
def save_model(self, filepath, include_metadata=True, compress=True):
    """
    Save the trained model to a file with optional metadata and compression.
    
    Args:
        filepath (str): Path where the model should be saved
        include_metadata (bool): Whether to include training metadata
        compress (bool): Whether to compress the model file
    
    Returns:
        dict: Save operation results including file size and checksum
        
    Example:
        >>> # Save with full metadata
        >>> result = recognizer.save_model(
        ...     'models/attendance_model_v2.yml',
        ...     include_metadata=True,
        ...     compress=True
        ... )
        >>> print(f"Model saved: {result['file_size']} bytes")
        >>> print(f"Checksum: {result['checksum']}")
    """
```

#### load_model()

```python
def load_model(self, filepath, verify_checksum=True, load_metadata=True):
    """
    Load a pre-trained model from file with integrity verification.
    
    Args:
        filepath (str): Path to the saved model file
        verify_checksum (bool): Whether to verify file integrity
        load_metadata (bool): Whether to load training metadata
    
    Returns:
        dict: Model loading results and metadata
        
    Raises:
        ModelLoadError: If model file is corrupted or incompatible
        
    Example:
        >>> # Load model with verification
        >>> result = recognizer.load_model(
        ...     'models/attendance_model_v2.yml',
        ...     verify_checksum=True
        ... )
        >>> 
        >>> print(f"Model loaded successfully")
        >>> print(f"Training date: {result['metadata']['training_date']}")
        >>> print(f"Samples used: {result['metadata']['total_samples']}")
    """
```

#### update_model()

```python
def update_model(self, new_face_samples, new_labels, retrain_method='incremental',
                preserve_old_data=True, validation_split=0.1):
    """
    Update existing model with new training samples using incremental learning.
    
    Args:
        new_face_samples (list): New face images to add
        new_labels (list): Corresponding labels for new samples
        retrain_method (str): Method for updating ('incremental', 'full_retrain')
        preserve_old_data (bool): Whether to keep original training data
        validation_split (float): Validation split for new data
    
    Returns:
        dict: Update results including accuracy changes
        
    Example:
        >>> # Add new person to existing model
        >>> new_person_faces = [face1, face2, face3, face4, face5]
        >>> new_person_labels = [101, 101, 101, 101, 101]  # New person ID
        >>> 
        >>> update_result = recognizer.update_model(
        ...     new_person_faces,
        ...     new_person_labels,
        ...     retrain_method='incremental'
        ... )
        >>> 
        >>> print(f"Model updated successfully")
        >>> print(f"Accuracy change: {update_result['accuracy_delta']:.3f}")
    """
```

### Advanced Recognition Methods

#### recognize_with_verification()

```python
def recognize_with_verification(self, face_image, verification_samples=None,
                              verification_threshold=0.8, use_ensemble=False):
    """
    Perform recognition with additional verification for high-security applications.
    
    Args:
        face_image (numpy.ndarray): Face image to recognize
        verification_samples (list, optional): Additional samples for verification
        verification_threshold (float): Threshold for verification step
        use_ensemble (bool): Whether to use ensemble of multiple models
    
    Returns:
        dict: Comprehensive recognition results with verification status
        
    Example:
        >>> # High-security recognition
        >>> result = recognizer.recognize_with_verification(
        ...     suspicious_face,
        ...     verification_threshold=0.9,
        ...     use_ensemble=True
        ... )
        >>> 
        >>> if result['verified']:
        ...     print(f"Identity verified: {result['label']}")
        ...     print(f"Confidence: {result['confidence']:.3f}")
        ...     print(f"Verification score: {result['verification_score']:.3f}")
        ... else:
        ...     print("Identity could not be verified")
    """
```

#### recognize_multiple_faces()

```python
def recognize_multiple_faces(self, face_regions, min_confidence=None,
                           deduplicate=True, spatial_grouping=False):
    """
    Recognize multiple faces in a single operation with advanced processing.
    
    Args:
        face_regions (list): List of face image regions
        min_confidence (float, optional): Minimum confidence for valid recognition
        deduplicate (bool): Whether to remove duplicate identifications
        spatial_grouping (bool): Whether to group nearby faces of same person
    
    Returns:
        list: Recognition results for all faces with additional metadata
        
    Example:
        >>> # Process group photo
        >>> faces = detector.detect_faces(group_image)
        >>> face_regions = []
        >>> 
        >>> for x, y, w, h in faces:
        ...     face_region = group_image[y:y+h, x:x+w]
        ...     face_regions.append(face_region)
        >>> 
        >>> results = recognizer.recognize_multiple_faces(
        ...     face_regions,
        ...     min_confidence=75,
        ...     deduplicate=True
        ... )
        >>> 
        >>> unique_people = set(r['label'] for r in results if r['recognized'])
        >>> print(f"Recognized {len(unique_people)} unique people")
    """
```

---

## LBPH Algorithm Details

### Algorithm Overview

The Local Binary Pattern Histogram (LBPH) algorithm works through the following steps:

1. **Local Binary Pattern Calculation**: Convert each pixel to a binary pattern based on neighboring pixels
2. **Grid Division**: Divide the face image into a grid of cells
3. **Histogram Generation**: Calculate histogram of LBP values for each cell
4. **Feature Vector Creation**: Concatenate all cell histograms into a single feature vector
5. **Distance Calculation**: Use Chi-square distance for comparing feature vectors

### Implementation Details

```python
class LBPHImplementation:
    """
    Detailed implementation of LBPH algorithm for educational purposes
    """
    
    def __init__(self, radius=1, neighbors=8):
        self.radius = radius
        self.neighbors = neighbors
        
    def calculate_lbp(self, image):
        """
        Calculate Local Binary Pattern for entire image
        
        Args:
            image (numpy.ndarray): Grayscale input image
            
        Returns:
            numpy.ndarray: LBP image with same dimensions
        """
        rows, cols = image.shape
        lbp_image = np.zeros((rows, cols), dtype=np.uint8)
        
        # Define neighbor positions in circular pattern
        angles = np.linspace(0, 2 * np.pi, self.neighbors, endpoint=False)
        neighbor_positions = []
        
        for angle in angles:
            dy = -self.radius * np.sin(angle)
            dx = self.radius * np.cos(angle)
            neighbor_positions.append((dy, dx))
        
        # Calculate LBP for each pixel
        for i in range(self.radius, rows - self.radius):
            for j in range(self.radius, cols - self.radius):
                center_pixel = image[i, j]
                lbp_value = 0
                
                # Compare with each neighbor
                for k, (dy, dx) in enumerate(neighbor_positions):
                    # Bilinear interpolation for non-integer positions
                    neighbor_value = self._bilinear_interpolate(
                        image, i + dy, j + dx
                    )
                    
                    # Set bit if neighbor >= center
                    if neighbor_value >= center_pixel:
                        lbp_value |= (1 << k)
                
                lbp_image[i, j] = lbp_value
        
        return lbp_image
    
    def _bilinear_interpolate(self, image, y, x):
        """Perform bilinear interpolation for non-integer coordinates"""
        x1, y1 = int(x), int(y)
        x2, y2 = x1 + 1, y1 + 1
        
        # Handle boundary conditions
        if x2 >= image.shape[1]:
            x2 = x1
        if y2 >= image.shape[0]:
            y2 = y1
            
        # Get surrounding pixel values
        q11 = image[y1, x1]
        q12 = image[y2, x1] if y2 < image.shape[0] else q11
        q21 = image[y1, x2] if x2 < image.shape[1] else q11
        q22 = image[y2, x2] if (y2 < image.shape[0] and x2 < image.shape[1]) else q11
        
        # Bilinear interpolation
        wx = x - x1
        wy = y - y1
        
        interpolated = (q11 * (1 - wx) * (1 - wy) +
                       q21 * wx * (1 - wy) +
                       q12 * (1 - wx) * wy +
                       q22 * wx * wy)
        
        return interpolated
    
    def create_histogram_grid(self, lbp_image, grid_x=8, grid_y=8):
        """
        Create histogram for each grid cell
        
        Args:
            lbp_image (numpy.ndarray): LBP processed image
            grid_x (int): Number of horizontal grid divisions
            grid_y (int): Number of vertical grid divisions
            
        Returns:
            numpy.ndarray: Concatenated histogram feature vector
        """
        rows, cols = lbp_image.shape
        cell_height = rows // grid_y
        cell_width = cols // grid_x
        
        histograms = []
        
        # Process each grid cell
        for i in range(grid_y):
            for j in range(grid_x):
                # Define cell boundaries
                y1 = i * cell_height
                y2 = min((i + 1) * cell_height, rows)
                x1 = j * cell_width
                x2 = min((j + 1) * cell_width, cols)
                
                # Extract cell region
                cell = lbp_image[y1:y2, x1:x2]
                
                # Calculate histogram (256 bins for 8-neighbor LBP)
                hist, _ = np.histogram(cell.ravel(), bins=256, range=(0, 256))
                
                # Normalize histogram
                hist = hist.astype(np.float32)
                hist = hist / (hist.sum() + 1e-7)  # Avoid division by zero
                
                histograms.append(hist)
        
        # Concatenate all histograms
        feature_vector = np.concatenate(histograms)
        return feature_vector
    
    def chi_square_distance(self, hist1, hist2):
        """
        Calculate Chi-square distance between two histograms
        
        Args:
            hist1, hist2 (numpy.ndarray): Histogram vectors to compare
            
        Returns:
            float: Chi-square distance (lower = more similar)
        """
        # Avoid division by zero
        denominator = hist1 + hist2
        denominator[denominator == 0] = 1e-10
        
        # Chi-square formula
        chi_square = 0.5 * np.sum(((hist1 - hist2) ** 2) / denominator)
        return chi_square

# Usage example
def lbph_algorithm_demo():
    """Demonstrate LBPH algorithm step by step"""
    
    # Load sample face image
    face_image = cv2.imread('sample_face.jpg', cv2.IMREAD_GRAYSCALE)
    face_image = cv2.resize(face_image, (100, 100))  # Standardize size
    
    # Initialize LBPH implementation
    lbph = LBPHImplementation(radius=1, neighbors=8)
    
    # Step 1: Calculate LBP
    print("Step 1: Calculating Local Binary Patterns...")
    lbp_image = lbph.calculate_lbp(face_image)
    
    # Step 2: Create histogram grid
    print("Step 2: Creating histogram grid...")
    feature_vector = lbph.create_histogram_grid(lbp_image, grid_x=8, grid_y=8)
    
    print(f"Feature vector length: {len(feature_vector)}")
    print(f"Feature vector shape: {feature_vector.shape}")
    
    # Visualize results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(face_image, cmap='gray')
    axes[0].set_title('Original Face')
    axes[0].axis('off')
    
    axes[1].imshow(lbp_image, cmap='gray')
    axes[1].set_title('LBP Image')
    axes[1].axis('off')
    
    axes[2].plot(feature_vector)
    axes[2].set_title('Feature Vector')
    axes[2].set_xlabel('Feature Index')
    axes[2].set_ylabel('Normalized Count')
    
    plt.tight_layout()
    plt.show()
    
    return feature_vector
```

---

## Training and Model Management

### Comprehensive Training Pipeline

```python
class AdvancedTrainingManager:
    """
    Advanced training manager with data augmentation, validation, and optimization
    """
    
    def __init__(self, recognizer):
        self.recognizer = recognizer
        self.training_history = []
        self.validation_metrics = {}
        
    def train_with_validation(self, face_samples, labels, config=None):
        """
        Train model with comprehensive validation and monitoring
        
        Args:
            face_samples (list): Training face images
            labels (list): Corresponding labels
            config (dict, optional): Training configuration
            
        Returns:
            dict: Comprehensive training results
        """
        if config is None:
            config = self._get_default_config()
        
        # Validate input data
        self._validate_training_data(face_samples, labels)
        
        # Preprocess data
        processed_samples, processed_labels = self._preprocess_training_data(
            face_samples, labels, config
        )
        
        # Split data
        train_samples, val_samples, train_labels, val_labels = self._split_data(
            processed_samples, processed_labels, config['validation_split']
        )
        
        # Apply data augmentation if enabled
        if config['enable_augmentation']:
            train_samples, train_labels = self._augment_data(
                train_samples, train_labels, config['augmentation_params']
            )
        
        # Train model
        training_start = time.time()
        
        self.recognizer.train(train_samples, train_labels)
        
        training_time = time.time() - training_start
        
        # Evaluate on validation set
        val_results = self._evaluate_model(val_samples, val_labels)
        
        # Calculate training metrics
        train_results = self._evaluate_model(train_samples, train_labels)
        
        # Compile comprehensive results
        results = {
            'training_accuracy': train_results['accuracy'],
            'validation_accuracy': val_results['accuracy'],
            'training_time': training_time,
            'samples_processed': len(train_samples),
            'validation_samples': len(val_samples),
            'confusion_matrix': val_results['confusion_matrix'],
            'per_class_metrics': val_results['per_class_metrics'],
            'model_parameters': self._get_model_parameters(),
            'training_config': config
        }
        
        # Store in history
        self.training_history.append(results)
        
        return results
    
    def _get_default_config(self):
        """Get default training configuration"""
        return {
            'validation_split': 0.2,
            'enable_augmentation': True,
            'augmentation_params': {
                'rotation_range': 10,
                'brightness_range': 0.2,
                'noise_level': 0.1,
                'blur_probability': 0.1
            },
            'preprocessing': {
                'normalize': True,
                'histogram_equalization': True,
                'gaussian_blur': (1, 1)
            }
        }
    
    def _validate_training_data(self, face_samples, labels):
        """Validate training data quality and consistency"""
        if len(face_samples) != len(labels):
            raise ValueError("Number of samples and labels must match")
        
        if len(face_samples) == 0:
            raise ValueError("No training samples provided")
        
        # Check for minimum samples per class
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        min_samples_per_class = 3
        insufficient_classes = [
            label for label, count in label_counts.items() 
            if count < min_samples_per_class
        ]
        
        if insufficient_classes:
            print(f"Warning: Classes with insufficient samples: {insufficient_classes}")
            print(f"Minimum recommended samples per class: {min_samples_per_class}")
        
        # Check image dimensions consistency
        if face_samples:
            reference_shape = face_samples[0].shape
            inconsistent_shapes = [
                i for i, sample in enumerate(face_samples)
                if sample.shape != reference_shape
            ]
            
            if inconsistent_shapes:
                print(f"Warning: Inconsistent image shapes found at indices: {inconsistent_shapes[:10]}")
    
    def _preprocess_training_data(self, face_samples, labels, config):
        """Apply preprocessing to training data"""
        processed_samples = []
        
        for sample in face_samples:
            processed_sample = sample.copy()
            
            # Convert to grayscale if needed
            if len(processed_sample.shape) == 3:
                processed_sample = cv2.cvtColor(processed_sample, cv2.COLOR_BGR2GRAY)
            
            # Apply preprocessing steps
            if config['preprocessing']['histogram_equalization']:
                processed_sample = cv2.equalizeHist(processed_sample)
            
            if config['preprocessing']['gaussian_blur']:
                kernel_size = config['preprocessing']['gaussian_blur']
                processed_sample = cv2.GaussianBlur(processed_sample, kernel_size, 0)
            
            if config['preprocessing']['normalize']:
                processed_sample = processed_sample.astype(np.float32) / 255.0
                processed_sample = (processed_sample * 255).astype(np.uint8)
            
            processed_samples.append(processed_sample)
        
        return processed_samples, labels
    
    def _augment_data(self, samples, labels, augmentation_params):
        """Apply data augmentation techniques"""
        augmented_samples = []
        augmented_labels = []
        
        for sample, label in zip(samples, labels):
            # Original sample
            augmented_samples.append(sample)
            augmented_labels.append(label)
            
            # Apply augmentations
            for _ in range(2):  # Generate 2 augmented versions per sample
                augmented_sample = self._apply_augmentation(sample, augmentation_params)
                augmented_samples.append(augmented_sample)
                augmented_labels.append(label)
        
        return augmented_samples, augmented_labels
    
    def _apply_augmentation(self, image, params):
        """Apply single augmentation to image"""
        augmented = image.copy()
        
        # Random rotation
        if params['rotation_range'] > 0:
            angle = np.random.uniform(-params['rotation_range'], params['rotation_range'])
            center = (image.shape[1] // 2, image.shape[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            augmented = cv2.warpAffine(augmented, rotation_matrix, 
                                     (image.shape[1], image.shape[0]))
        
        # Random brightness adjustment
        if params['brightness_range'] > 0:
            brightness_factor = np.random.uniform(
                1 - params['brightness_range'], 
                1 + params['brightness_range']
            )
            augmented = np.clip(augmented * brightness_factor, 0, 255).astype(np.uint8)
        
        # Add random noise
        if params['noise_level'] > 0:
            noise = np.random.normal(0, params['noise_level'] * 255, augmented.shape)
            augmented = np.clip(augmented + noise, 0, 255).astype(np.uint8)
        
        # Random blur
        if np.random.random() < params['blur_probability']:
            kernel_size = np.random.choice([3, 5])
            augmented = cv2.GaussianBlur(augmented, (kernel_size, kernel_size), 0)
        
        return augmented
    
    def _evaluate_model(self, test_samples, test_labels):
        """Evaluate model performance on test set"""
        predictions = []
        confidences = []
        
        for sample in test_samples:
            label, confidence = self.recognizer.predict(sample)
            predictions.append(label)
            confidences.append(confidence)
        
        # Calculate accuracy
        correct_predictions = sum(1 for pred, true in zip(predictions, test_labels) 
                                if pred == true)
        accuracy = correct_predictions / len(test_labels)
        
        # Create confusion matrix
        unique_labels = sorted(list(set(test_labels)))
        confusion_matrix = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)
        
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        
        for pred, true in zip(predictions, test_labels):
            if pred in label_to_idx and true in label_to_idx:
                confusion_matrix[label_to_idx[true], label_to_idx[pred]] += 1
        
        # Calculate per-class metrics
        per_class_metrics = {}
        for i, label in enumerate(unique_labels):
            true_positives = confusion_matrix[i, i]
            false_positives = confusion_matrix[:, i].sum() - true_positives
            false_negatives = confusion_matrix[i, :].sum() - true_positives
            
            precision = true_positives / (true_positives + false_positives + 1e-7)
            recall = true_positives / (true_positives + false_negatives + 1e-7)
            f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)
            
            per_class_metrics[label] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'support': confusion_matrix[i, :].sum()
            }
        
        return {
            'accuracy': accuracy,
            'predictions': predictions,
            'confidences': confidences,
            'confusion_matrix': confusion_matrix,
            'per_class_metrics': per_class_metrics,
            'unique_labels': unique_labels
        }

# Usage example
def comprehensive_training_example():
    """Example of comprehensive model training"""
    
    # Initialize recognizer and training manager
    recognizer = FaceRecognizer(radius=1, neighbors=8, grid_x=8, grid_y=8)
    trainer = AdvancedTrainingManager(recognizer)
    
    # Load training data (example)
    face_samples, labels = load_training_dataset('training_data/')
    
    # Configure training
    config = {
        'validation_split': 0.25,
        'enable_augmentation': True,
        'augmentation_params': {
            'rotation_range': 15,
            'brightness_range': 0.3,
            'noise_level': 0.05,
            'blur_probability': 0.15
        }
    }
    
    # Train model
    results = trainer.train_with_validation(face_samples, labels, config)
    
    # Print results
    print("Training Results:")
    print(f"Training Accuracy: {results['training_accuracy']:.3f}")
    print(f"Validation Accuracy: {results['validation_accuracy']:.3f}")
    print(f"Training Time: {results['training_time']:.2f} seconds")
    print(f"Samples Processed: {results['samples_processed']}")
    
    # Print per-class metrics
    print("\nPer-Class Performance:")
    for label, metrics in results['per_class_metrics'].items():
        print(f"Class {label}:")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  F1-Score: {metrics['f1_score']:.3f}")
        print(f"  Support: {metrics['support']}")
    
    # Save model
    recognizer.save_model('models/comprehensive_model.yml')
    
    return results
```

---

## Performance Optimization

### Multi-threaded Recognition

```python
import concurrent.futures
import threading
from queue import Queue

class OptimizedFaceRecognizer:
    """
    Optimized face recognizer with multi-threading and caching capabilities
    """
    
    def __init__(self, base_recognizer, max_workers=4, enable_caching=True):
        self.base_recognizer = base_recognizer
        self.max_workers = max_workers
        self.enable_caching = enable_caching
        
        # Feature cache for improved performance
        self.feature_cache = {} if enable_caching else None
        self.cache_lock = threading.Lock() if enable_caching else None
        
        # Performance monitoring
        self.recognition_times = []
        self.cache_hits = 0
        self.cache_misses = 0
    
    def predict_parallel(self, face_images, batch_size=None):
        """
        Perform parallel face recognition on multiple images
        
        Args:
            face_images (list): List of face images to recognize
            batch_size (int, optional): Size of processing batches
            
        Returns:
            list: Recognition results for all images
        """
        if batch_size is None:
            batch_size = max(1, len(face_images) // self.max_workers)
        
        results = [None] * len(face_images)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit batch processing tasks
            futures = []
            
            for i in range(0, len(face_images), batch_size):
                batch_end = min(i + batch_size, len(face_images))
                batch_images = face_images[i:batch_end]
                batch_indices = list(range(i, batch_end))
                
                future = executor.submit(self._process_batch, batch_images, batch_indices)
                futures.append(future)
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                batch_results = future.result()
                for idx, result in batch_results:
                    results[idx] = result
        
        return results
    
    def _process_batch(self, batch_images, batch_indices):
        """Process a batch of images in single thread"""
        batch_results = []
        
        for image, idx in zip(batch_images, batch_indices):
            start_time = time.time()
            
            # Check cache first
            if self.enable_caching:
                cache_key = self._generate_cache_key(image)
                cached_result = self._get_from_cache(cache_key)
                
                if cached_result is not None:
                    self.cache_hits += 1
                    batch_results.append((idx, cached_result))
                    continue
                else:
                    self.cache_misses += 1
            
            # Perform recognition
            label, confidence = self.base_recognizer.predict(image)
            result = {'label': label, 'confidence': confidence}
            
            # Cache result
            if self.enable_caching:
                self._store_in_cache(cache_key, result)
            
            # Track performance
            recognition_time = time.time() - start_time
            self.recognition_times.append(recognition_time)
            
            batch_results.append((idx, result))
        
        return batch_results
    
    def _generate_cache_key(self, image):
        """Generate cache key for image"""
        # Use image hash for cache key
        image_bytes = image.tobytes()
        return hash(image_bytes) % (10**8)  # Limit hash size
    
    def _get_from_cache(self, cache_key):
        """Retrieve result from cache"""
        with self.cache_lock:
            return self.feature_cache.get(cache_key)
    
    def _store_in_cache(self, cache_key, result):
        """Store result in cache"""
        with self.cache_lock:
            # Implement LRU cache with size limit
            max_cache_size = 1000
            
            if len(self.feature_cache) >= max_cache_size:
                # Remove oldest entry (simple FIFO for demonstration)
                oldest_key = next(iter(self.feature_cache))
                del self.feature_cache[oldest_key]
            
            self.feature_cache[cache_key] = result
    
    def get_performance_stats(self):
        """Get performance statistics"""
        stats = {
            'total_recognitions': len(self.recognition_times),
            'avg_recognition_time': np.mean(self.recognition_times) if self.recognition_times else 0,
            'min_recognition_time': np.min(self.recognition_times) if self.recognition_times else 0,
            'max_recognition_time': np.max(self.recognition_times) if self.recognition_times else 0,
        }
        
        if self.enable_caching:
            total_requests = self.cache_hits + self.cache_misses
            cache_hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
            
            stats.update({
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'cache_hit_rate': cache_hit_rate,
                'cache_size': len(self.feature_cache) if self.feature_cache else 0
            })
        
        return stats
    
    def clear_cache(self):
        """Clear the feature cache"""
        if self.enable_caching:
            with self.cache_lock:
                self.feature_cache.clear()
                self.cache_hits = 0
                self.cache_misses = 0

# Usage example
def optimized_recognition_example():
    """Example of optimized face recognition"""
    
    # Initialize base recognizer
    base_recognizer = FaceRecognizer()
    base_recognizer.load_model('models/attendance_model.yml')
    
    # Create optimized recognizer
    optimized_recognizer = OptimizedFaceRecognizer(
        base_recognizer,
        max_workers=8,
        enable_caching=True
    )
    
    # Load test images
    test_images = []
    for i in range(100):
        img_path = f'test_faces/face_{i:03d}.jpg'
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            test_images.append(img)
    
    # Perform parallel recognition
    print(f"Processing {len(test_images)} images...")
    start_time = time.time()
    
    results = optimized_recognizer.predict_parallel(test_images, batch_size=10)
    
    total_time = time.time() - start_time
    
    # Print results
    recognized_count = sum(1 for r in results if r['confidence'] < 100)
    print(f"Recognized {recognized_count}/{len(results)} faces")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per image: {total_time/len(results):.3f} seconds")
    
    # Print performance stats
    stats = optimized_recognizer.get_performance_stats()
    print("\nPerformance Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
```

This comprehensive Face Recognition API documentation provides detailed information about the LBPH-based recognition system, including advanced training techniques, performance optimization, and practical usage examples. The documentation covers both theoretical aspects of the algorithm and practical implementation details for real-world deployment.