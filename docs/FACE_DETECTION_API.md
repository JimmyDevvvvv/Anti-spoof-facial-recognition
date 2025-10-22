# Face Detection API Documentation

## Overview

The Face Detection API provides robust face detection capabilities using Haar Cascade classifiers. This module is optimized for real-time face detection in various lighting conditions and can handle multiple faces in a single image.

## Table of Contents
1. [FaceDetector Class](#facedetector-class)
2. [Detection Algorithms](#detection-algorithms)
3. [Performance Optimization](#performance-optimization)
4. [Quality Assessment](#quality-assessment)
5. [Real-time Processing](#real-time-processing)
6. [Advanced Examples](#advanced-examples)

---

## FaceDetector Class

### Class Definition

```python
class FaceDetector:
    """
    A comprehensive face detection system using Haar Cascade classifiers.
    
    This class provides methods for detecting faces in images, validating face quality,
    and optimizing detection parameters for different scenarios.
    """
    
    def __init__(self, cascade_path='haarcascade_frontalface_default.xml', 
                 scale_factor=1.1, min_neighbors=5, min_size=(30, 30),
                 max_size=(), flags=cv2.CASCADE_SCALE_IMAGE):
        """
        Initialize the FaceDetector with specified parameters.
        
        Args:
            cascade_path (str): Path to Haar cascade XML file
            scale_factor (float): How much the image size is reduced at each scale (1.05-1.4)
            min_neighbors (int): How many neighbors each face rectangle should have (3-6)
            min_size (tuple): Minimum possible face size in pixels
            max_size (tuple): Maximum possible face size in pixels (empty = no limit)
            flags (int): OpenCV cascade flags for detection optimization
        """
```

### Core Methods

#### detect_faces()

```python
def detect_faces(self, image, return_gray=False, return_confidence=False,
                scale_factor=None, min_neighbors=None):
    """
    Detect faces in the provided image using Haar Cascade classifier.
    
    Args:
        image (numpy.ndarray): Input image in BGR format
        return_gray (bool): Whether to return grayscale conversion
        return_confidence (bool): Whether to return detection confidence scores
        scale_factor (float, optional): Override default scale factor
        min_neighbors (int, optional): Override default min neighbors
    
    Returns:
        tuple: Depending on parameters:
            - faces (list): List of (x, y, width, height) tuples
            - gray_image (numpy.ndarray, optional): Grayscale image
            - confidences (list, optional): Confidence scores for each detection
    
    Raises:
        FaceDetectionError: If image is invalid or detection fails
        
    Example:
        >>> detector = FaceDetector()
        >>> image = cv2.imread('group_photo.jpg')
        >>> faces = detector.detect_faces(image)
        >>> print(f"Found {len(faces)} faces")
        
        >>> # With additional return values
        >>> faces, gray, confidences = detector.detect_faces(
        ...     image, return_gray=True, return_confidence=True
        ... )
    """
```

#### detect_faces_multiscale()

```python
def detect_faces_multiscale(self, image, scale_factors=[1.05, 1.1, 1.2],
                           min_neighbors_list=[3, 5, 7], merge_overlapping=True):
    """
    Perform multi-scale face detection with different parameters for improved accuracy.
    
    Args:
        image (numpy.ndarray): Input image
        scale_factors (list): List of scale factors to try
        min_neighbors_list (list): List of min_neighbors values to try
        merge_overlapping (bool): Whether to merge overlapping detections
    
    Returns:
        list: Combined face detections with confidence scores
        
    Example:
        >>> faces = detector.detect_faces_multiscale(
        ...     image,
        ...     scale_factors=[1.05, 1.1, 1.15],
        ...     min_neighbors_list=[3, 4, 5]
        ... )
    """
```

#### validate_face()

```python
def validate_face(self, face_region, min_quality_threshold=0.7,
                 check_blur=True, check_brightness=True, check_contrast=True):
    """
    Validate the quality of a detected face region using multiple metrics.
    
    Args:
        face_region (numpy.ndarray): Cropped face image
        min_quality_threshold (float): Minimum acceptable quality score (0.0-1.0)
        check_blur (bool): Whether to check for image blur
        check_brightness (bool): Whether to check brightness levels
        check_contrast (bool): Whether to check contrast levels
    
    Returns:
        dict: Validation results containing:
            - is_valid (bool): Overall validation result
            - quality_score (float): Combined quality score
            - blur_score (float): Blur detection score
            - brightness_score (float): Brightness adequacy score
            - contrast_score (float): Contrast adequacy score
            - recommendations (list): Improvement suggestions
    
    Example:
        >>> x, y, w, h = faces[0]
        >>> face_region = image[y:y+h, x:x+w]
        >>> validation = detector.validate_face(face_region)
        >>> 
        >>> if validation['is_valid']:
        ...     print(f"Face quality acceptable: {validation['quality_score']:.2f}")
        ... else:
        ...     print("Quality issues:", validation['recommendations'])
    """
```

#### draw_faces()

```python
def draw_faces(self, image, faces, color=(0, 255, 0), thickness=2,
              show_confidence=False, confidences=None, font_scale=0.5):
    """
    Draw rectangles around detected faces with optional confidence scores.
    
    Args:
        image (numpy.ndarray): Input image
        faces (list): List of face coordinates [(x, y, w, h), ...]
        color (tuple): BGR color for rectangles
        thickness (int): Rectangle line thickness
        show_confidence (bool): Whether to display confidence scores
        confidences (list, optional): Confidence scores for each face
        font_scale (float): Font size for confidence text
    
    Returns:
        numpy.ndarray: Annotated image with face rectangles
        
    Example:
        >>> annotated = detector.draw_faces(
        ...     image, faces, 
        ...     color=(255, 0, 0),  # Red rectangles
        ...     show_confidence=True,
        ...     confidences=confidence_scores
        ... )
        >>> cv2.imshow('Detected Faces', annotated)
    """
```

### Advanced Methods

#### detect_faces_roi()

```python
def detect_faces_roi(self, image, roi_regions, adaptive_params=True):
    """
    Detect faces within specific regions of interest for improved performance.
    
    Args:
        image (numpy.ndarray): Input image
        roi_regions (list): List of (x, y, width, height) regions to search
        adaptive_params (bool): Whether to adapt detection parameters per ROI
    
    Returns:
        dict: Faces detected in each ROI region
        
    Example:
        >>> # Define ROIs (e.g., from motion detection)
        >>> rois = [(100, 100, 300, 300), (500, 200, 200, 200)]
        >>> roi_faces = detector.detect_faces_roi(image, rois)
        >>> 
        >>> for roi_idx, faces in roi_faces.items():
        ...     print(f"ROI {roi_idx}: {len(faces)} faces")
    """
```

#### track_faces()

```python
def track_faces(self, previous_faces, current_image, max_displacement=50):
    """
    Track faces across consecutive frames using position prediction.
    
    Args:
        previous_faces (list): Faces from previous frame
        current_image (numpy.ndarray): Current frame
        max_displacement (int): Maximum expected face movement in pixels
    
    Returns:
        dict: Tracking results with face IDs and positions
        
    Example:
        >>> # Initialize tracking
        >>> tracker_data = {}
        >>> 
        >>> # In video loop
        >>> for frame in video_frames:
        ...     if 'previous_faces' in tracker_data:
        ...         tracking_result = detector.track_faces(
        ...             tracker_data['previous_faces'], frame
        ...         )
        ...         tracker_data.update(tracking_result)
        ...     else:
        ...         faces = detector.detect_faces(frame)
        ...         tracker_data['previous_faces'] = faces
    """
```

---

## Detection Algorithms

### Haar Cascade Algorithm

The Haar Cascade algorithm uses machine learning to detect objects in images through the following process:

1. **Feature Extraction**: Uses Haar-like features (rectangular patterns) to identify facial characteristics
2. **Integral Image**: Computes integral image for fast feature calculation  
3. **AdaBoost Training**: Selects most discriminative features from large set
4. **Cascade Structure**: Organizes classifiers in stages for efficient rejection of non-faces

#### Haar Features Used

```python
# Common Haar features for face detection
HAAR_FEATURES = {
    'edge_features': 'Detect edges around eyes, nose, mouth',
    'line_features': 'Detect horizontal/vertical lines in facial structure', 
    'center_surround': 'Detect darker eye regions vs lighter cheek regions',
    'diagonal_features': 'Detect diagonal patterns in facial geometry'
}
```

### Detection Pipeline

```python
def detection_pipeline_example():
    """
    Example of the complete face detection pipeline
    """
    detector = FaceDetector()
    
    # Step 1: Load and preprocess image
    image = cv2.imread('input.jpg')
    if image is None:
        raise ValueError("Could not load image")
    
    # Step 2: Convert to grayscale (Haar cascades work on grayscale)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Step 3: Histogram equalization for better contrast
    gray = cv2.equalizeHist(gray)
    
    # Step 4: Detect faces
    faces = detector.detect_faces(image)
    
    # Step 5: Validate each detected face
    valid_faces = []
    for face in faces:
        x, y, w, h = face
        face_region = gray[y:y+h, x:x+w]
        validation = detector.validate_face(face_region)
        
        if validation['is_valid']:
            valid_faces.append(face)
    
    return valid_faces
```

---

## Performance Optimization

### Parameter Tuning

```python
class OptimizedFaceDetector(FaceDetector):
    """
    Optimized face detector with scenario-specific parameter sets
    """
    
    def __init__(self):
        super().__init__()
        
        # Parameter sets for different scenarios
        self.param_sets = {
            'real_time': {
                'scale_factor': 1.2,
                'min_neighbors': 3,
                'min_size': (40, 40)
            },
            'high_accuracy': {
                'scale_factor': 1.05,
                'min_neighbors': 7,
                'min_size': (30, 30)
            },
            'distant_faces': {
                'scale_factor': 1.1,
                'min_neighbors': 4,
                'min_size': (20, 20)
            },
            'group_photos': {
                'scale_factor': 1.1,
                'min_neighbors': 5,
                'min_size': (25, 25)
            }
        }
    
    def detect_optimized(self, image, scenario='real_time'):
        """
        Detect faces using optimized parameters for specific scenarios
        """
        params = self.param_sets.get(scenario, self.param_sets['real_time'])
        
        return self.detect_faces(
            image,
            scale_factor=params['scale_factor'],
            min_neighbors=params['min_neighbors']
        )
```

### Multi-threading Support

```python
import threading
from concurrent.futures import ThreadPoolExecutor
import queue

class ThreadedFaceDetector:
    """
    Thread-safe face detector for processing multiple images simultaneously
    """
    
    def __init__(self, max_workers=4):
        self.detector = FaceDetector()
        self.max_workers = max_workers
        self.result_queue = queue.Queue()
    
    def detect_batch(self, images, callback=None):
        """
        Process multiple images in parallel
        
        Args:
            images (list): List of images to process
            callback (function): Optional callback for each result
        
        Returns:
            list: Detection results for all images
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all detection tasks
            future_to_image = {
                executor.submit(self._detect_single, img, idx): idx 
                for idx, img in enumerate(images)
            }
            
            # Collect results
            for future in future_to_image:
                try:
                    result = future.result()
                    results.append(result)
                    
                    if callback:
                        callback(result)
                        
                except Exception as e:
                    print(f"Detection failed for image {future_to_image[future]}: {e}")
                    results.append({'faces': [], 'error': str(e)})
        
        return results
    
    def _detect_single(self, image, image_idx):
        """Single image detection with error handling"""
        try:
            faces = self.detector.detect_faces(image)
            return {
                'image_idx': image_idx,
                'faces': faces,
                'face_count': len(faces),
                'timestamp': time.time()
            }
        except Exception as e:
            return {
                'image_idx': image_idx,
                'faces': [],
                'error': str(e)
            }

# Usage example
def batch_processing_example():
    detector = ThreadedFaceDetector(max_workers=8)
    
    # Load multiple images
    images = []
    for i in range(10):
        img = cv2.imread(f'image_{i}.jpg')
        if img is not None:
            images.append(img)
    
    # Process all images
    results = detector.detect_batch(images)
    
    # Print results
    total_faces = sum(r['face_count'] for r in results if 'face_count' in r)
    print(f"Total faces detected across {len(images)} images: {total_faces}")
```

---

## Quality Assessment

### Comprehensive Quality Metrics

```python
class FaceQualityAssessor:
    """
    Advanced face quality assessment using multiple metrics
    """
    
    def __init__(self):
        self.blur_threshold = 100.0
        self.brightness_range = (50, 200)
        self.contrast_threshold = 50.0
        self.symmetry_threshold = 0.8
    
    def assess_comprehensive_quality(self, face_region):
        """
        Perform comprehensive quality assessment of a face region
        
        Returns:
            dict: Detailed quality assessment results
        """
        metrics = {}
        
        # Convert to grayscale if needed
        if len(face_region.shape) == 3:
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        else:
            gray_face = face_region
        
        # 1. Blur Detection (Laplacian variance)
        metrics['blur_score'] = self._calculate_blur_score(gray_face)
        metrics['is_sharp'] = metrics['blur_score'] > self.blur_threshold
        
        # 2. Brightness Assessment
        metrics['brightness_score'] = self._calculate_brightness_score(gray_face)
        metrics['brightness_adequate'] = (
            self.brightness_range[0] <= metrics['brightness_score'] <= self.brightness_range[1]
        )
        
        # 3. Contrast Assessment
        metrics['contrast_score'] = self._calculate_contrast_score(gray_face)
        metrics['contrast_adequate'] = metrics['contrast_score'] > self.contrast_threshold
        
        # 4. Face Symmetry
        metrics['symmetry_score'] = self._calculate_symmetry_score(gray_face)
        metrics['symmetry_adequate'] = metrics['symmetry_score'] > self.symmetry_threshold
        
        # 5. Eye Detection (for face completeness)
        metrics['eyes_detected'] = self._detect_eyes_in_face(gray_face)
        
        # 6. Pose Estimation
        metrics['pose_angle'] = self._estimate_face_pose(gray_face)
        metrics['frontal_pose'] = abs(metrics['pose_angle']) < 30  # degrees
        
        # Overall quality score
        quality_factors = [
            metrics['is_sharp'],
            metrics['brightness_adequate'], 
            metrics['contrast_adequate'],
            metrics['symmetry_adequate'],
            metrics['eyes_detected'],
            metrics['frontal_pose']
        ]
        
        metrics['overall_quality'] = sum(quality_factors) / len(quality_factors)
        metrics['is_high_quality'] = metrics['overall_quality'] > 0.7
        
        # Recommendations for improvement
        metrics['recommendations'] = self._generate_recommendations(metrics)
        
        return metrics
    
    def _calculate_blur_score(self, gray_image):
        """Calculate blur score using Laplacian variance"""
        return cv2.Laplacian(gray_image, cv2.CV_64F).var()
    
    def _calculate_brightness_score(self, gray_image):
        """Calculate average brightness"""
        return np.mean(gray_image)
    
    def _calculate_contrast_score(self, gray_image):
        """Calculate image contrast using standard deviation"""
        return np.std(gray_image)
    
    def _calculate_symmetry_score(self, gray_image):
        """Calculate face symmetry by comparing left and right halves"""
        h, w = gray_image.shape
        left_half = gray_image[:, :w//2]
        right_half = cv2.flip(gray_image[:, w//2:], 1)  # Flip horizontally
        
        # Resize to same dimensions if needed
        if left_half.shape != right_half.shape:
            min_width = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_width]
            right_half = right_half[:, :min_width]
        
        # Calculate similarity using normalized cross-correlation
        correlation = cv2.matchTemplate(left_half, right_half, cv2.TM_CCOEFF_NORMED)
        return float(correlation.max())
    
    def _detect_eyes_in_face(self, gray_face):
        """Detect if both eyes are visible in the face"""
        eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(gray_face, 1.1, 5)
        return len(eyes) >= 2
    
    def _estimate_face_pose(self, gray_face):
        """Estimate face pose angle (simplified)"""
        # This is a simplified pose estimation
        # In practice, you might use more sophisticated methods
        h, w = gray_face.shape
        
        # Compare left and right brightness
        left_brightness = np.mean(gray_face[:, :w//3])
        right_brightness = np.mean(gray_face[:, 2*w//3:])
        
        # Rough angle estimation based on lighting difference
        brightness_diff = left_brightness - right_brightness
        estimated_angle = brightness_diff * 0.5  # Simple scaling
        
        return np.clip(estimated_angle, -45, 45)
    
    def _generate_recommendations(self, metrics):
        """Generate improvement recommendations based on quality metrics"""
        recommendations = []
        
        if not metrics['is_sharp']:
            recommendations.append("Image is blurry - ensure camera is focused")
        
        if not metrics['brightness_adequate']:
            if metrics['brightness_score'] < self.brightness_range[0]:
                recommendations.append("Image is too dark - improve lighting")
            else:
                recommendations.append("Image is too bright - reduce lighting")
        
        if not metrics['contrast_adequate']:
            recommendations.append("Low contrast - adjust lighting or camera settings")
        
        if not metrics['symmetry_adequate']:
            recommendations.append("Face appears asymmetric - check face positioning")
        
        if not metrics['eyes_detected']:
            recommendations.append("Eyes not clearly visible - ensure face is fully visible")
        
        if not metrics['frontal_pose']:
            recommendations.append("Face not facing camera directly - adjust pose")
        
        return recommendations

# Usage example
def quality_assessment_example():
    detector = FaceDetector()
    assessor = FaceQualityAssessor()
    
    image = cv2.imread('test_face.jpg')
    faces = detector.detect_faces(image)
    
    for i, (x, y, w, h) in enumerate(faces):
        face_region = image[y:y+h, x:x+w]
        quality_report = assessor.assess_comprehensive_quality(face_region)
        
        print(f"\nFace {i+1} Quality Report:")
        print(f"Overall Quality: {quality_report['overall_quality']:.2f}")
        print(f"High Quality: {quality_report['is_high_quality']}")
        
        if quality_report['recommendations']:
            print("Recommendations:")
            for rec in quality_report['recommendations']:
                print(f"  - {rec}")
```

---

## Real-time Processing

### Video Stream Processing

```python
class RealTimeFaceDetector:
    """
    Real-time face detector optimized for video streams
    """
    
    def __init__(self, camera_index=0, target_fps=30):
        self.detector = FaceDetector(
            scale_factor=1.2,  # Faster detection
            min_neighbors=3,   # Fewer false positives filtering
            min_size=(40, 40)  # Reasonable minimum size
        )
        
        self.camera_index = camera_index
        self.target_fps = target_fps
        self.frame_skip = 1  # Process every nth frame
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.detection_times = []
        
    def start_detection_stream(self, callback=None, display=True):
        """
        Start real-time face detection from camera stream
        
        Args:
            callback (function): Function to call with detection results
            display (bool): Whether to display video window
        """
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames for performance if needed
                if frame_count % self.frame_skip == 0:
                    start_time = time.time()
                    
                    # Detect faces
                    faces = self.detector.detect_faces(frame)
                    
                    # Track detection time
                    detection_time = time.time() - start_time
                    self.detection_times.append(detection_time)
                    
                    # Call callback if provided
                    if callback:
                        callback(frame, faces, detection_time)
                    
                    # Display if requested
                    if display:
                        annotated_frame = self.detector.draw_faces(frame, faces)
                        
                        # Add FPS counter
                        fps = self._calculate_fps()
                        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        cv2.imshow('Real-time Face Detection', annotated_frame)
                
                frame_count += 1
                
                # Break on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def _calculate_fps(self):
        """Calculate current FPS"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:  # Update every second
            fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
            return fps
        return 0
    
    def get_performance_stats(self):
        """Get performance statistics"""
        if not self.detection_times:
            return {}
        
        return {
            'avg_detection_time': np.mean(self.detection_times),
            'min_detection_time': np.min(self.detection_times),
            'max_detection_time': np.max(self.detection_times),
            'std_detection_time': np.std(self.detection_times),
            'total_detections': len(self.detection_times)
        }

# Usage example
def real_time_detection_example():
    detector = RealTimeFaceDetector(camera_index=0, target_fps=30)
    
    def detection_callback(frame, faces, detection_time):
        print(f"Detected {len(faces)} faces in {detection_time:.3f}s")
        
        # Log face positions
        for i, (x, y, w, h) in enumerate(faces):
            print(f"  Face {i+1}: ({x}, {y}) - {w}x{h}")
    
    # Start detection
    detector.start_detection_stream(callback=detection_callback)
    
    # Print performance stats after stopping
    stats = detector.get_performance_stats()
    print("\nPerformance Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.3f}")
```

---

## Advanced Examples

### Multi-Cascade Detection

```python
class MultiCascadeFaceDetector:
    """
    Face detector using multiple cascade classifiers for improved accuracy
    """
    
    def __init__(self):
        self.cascades = {
            'frontal': cv2.CascadeClassifier('haarcascade_frontalface_default.xml'),
            'profile': cv2.CascadeClassifier('haarcascade_profileface.xml'),
            'alt': cv2.CascadeClassifier('haarcascade_frontalface_alt.xml'),
            'alt2': cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
        }
        
        # Remove any cascades that failed to load
        self.cascades = {k: v for k, v in self.cascades.items() if not v.empty()}
    
    def detect_faces_multi_cascade(self, image, merge_threshold=0.3):
        """
        Detect faces using multiple cascade classifiers and merge results
        
        Args:
            image (numpy.ndarray): Input image
            merge_threshold (float): Threshold for merging overlapping detections
        
        Returns:
            list: Merged face detections with confidence scores
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        all_detections = []
        
        # Run detection with each cascade
        for cascade_name, cascade in self.cascades.items():
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Add cascade name to each detection
            for face in faces:
                x, y, w, h = face
                all_detections.append({
                    'bbox': (x, y, w, h),
                    'cascade': cascade_name,
                    'confidence': 1.0  # Default confidence
                })
        
        # Merge overlapping detections
        merged_faces = self._merge_overlapping_detections(all_detections, merge_threshold)
        
        return merged_faces
    
    def _merge_overlapping_detections(self, detections, threshold):
        """Merge overlapping face detections"""
        if not detections:
            return []
        
        # Convert to format suitable for Non-Maximum Suppression
        boxes = []
        scores = []
        
        for det in detections:
            x, y, w, h = det['bbox']
            boxes.append([x, y, x + w, y + h])
            scores.append(det['confidence'])
        
        boxes = np.array(boxes)
        scores = np.array(scores)
        
        # Apply Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), 
            scores.tolist(), 
            score_threshold=0.5,
            nms_threshold=threshold
        )
        
        merged_faces = []
        if len(indices) > 0:
            for i in indices.flatten():
                x1, y1, x2, y2 = boxes[i]
                merged_faces.append({
                    'bbox': (x1, y1, x2 - x1, y2 - y1),
                    'confidence': scores[i]
                })
        
        return merged_faces

# Usage example
def multi_cascade_example():
    detector = MultiCascadeFaceDetector()
    
    image = cv2.imread('challenging_faces.jpg')
    faces = detector.detect_faces_multi_cascade(image)
    
    print(f"Detected {len(faces)} faces using multiple cascades")
    
    # Draw results
    for face in faces:
        x, y, w, h = face['bbox']
        confidence = face['confidence']
        
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"{confidence:.2f}", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow('Multi-Cascade Detection', image)
    cv2.waitKey(0)
```

### Adaptive Detection Parameters

```python
class AdaptiveFaceDetector:
    """
    Face detector that adapts parameters based on image characteristics
    """
    
    def __init__(self):
        self.base_detector = FaceDetector()
        
    def detect_faces_adaptive(self, image):
        """
        Detect faces with parameters adapted to image characteristics
        """
        # Analyze image characteristics
        image_stats = self._analyze_image(image)
        
        # Adapt parameters based on analysis
        params = self._adapt_parameters(image_stats)
        
        # Detect with adapted parameters
        faces = self.base_detector.detect_faces(
            image,
            scale_factor=params['scale_factor'],
            min_neighbors=params['min_neighbors']
        )
        
        return faces, image_stats, params
    
    def _analyze_image(self, image):
        """Analyze image characteristics for parameter adaptation"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        stats = {
            'brightness': np.mean(gray),
            'contrast': np.std(gray),
            'resolution': gray.shape,
            'noise_level': self._estimate_noise(gray),
            'edge_density': self._calculate_edge_density(gray)
        }
        
        return stats
    
    def _adapt_parameters(self, image_stats):
        """Adapt detection parameters based on image statistics"""
        params = {
            'scale_factor': 1.1,
            'min_neighbors': 5,
            'min_size': (30, 30)
        }
        
        # Adapt scale factor based on resolution
        if image_stats['resolution'][0] > 1000:  # High resolution
            params['scale_factor'] = 1.05  # More thorough search
        elif image_stats['resolution'][0] < 400:  # Low resolution
            params['scale_factor'] = 1.2   # Faster search
        
        # Adapt min_neighbors based on noise
        if image_stats['noise_level'] > 20:  # Noisy image
            params['min_neighbors'] = 7     # More strict filtering
        elif image_stats['noise_level'] < 5:  # Clean image
            params['min_neighbors'] = 3     # Less strict filtering
        
        # Adapt min_size based on resolution
        resolution_area = image_stats['resolution'][0] * image_stats['resolution'][1]
        if resolution_area > 1000000:  # Large image
            params['min_size'] = (50, 50)
        elif resolution_area < 100000:  # Small image
            params['min_size'] = (20, 20)
        
        return params
    
    def _estimate_noise(self, gray_image):
        """Estimate noise level in image"""
        # Use Laplacian to estimate noise
        laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
        return min(laplacian_var / 1000, 100)  # Normalize to 0-100 scale
    
    def _calculate_edge_density(self, gray_image):
        """Calculate edge density in image"""
        edges = cv2.Canny(gray_image, 50, 150)
        return np.sum(edges > 0) / edges.size

# Usage example
def adaptive_detection_example():
    detector = AdaptiveFaceDetector()
    
    # Test with different types of images
    test_images = [
        'high_res_group.jpg',
        'low_light_selfie.jpg',
        'noisy_webcam.jpg',
        'professional_portrait.jpg'
    ]
    
    for img_path in test_images:
        image = cv2.imread(img_path)
        if image is not None:
            faces, stats, params = detector.detect_faces_adaptive(image)
            
            print(f"\n{img_path}:")
            print(f"  Image stats: {stats}")
            print(f"  Adapted params: {params}")
            print(f"  Faces detected: {len(faces)}")
```

This comprehensive Face Detection API documentation provides detailed information about all the public interfaces, methods, and advanced usage patterns for the face detection component of the attendance system. Each example includes proper error handling and demonstrates real-world usage scenarios.