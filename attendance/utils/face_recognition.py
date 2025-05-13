import cv2
import numpy as np
import pickle
import os
import logging
from pathlib import Path
from django.conf import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if dlib and face_recognition are available, otherwise use OpenCV's face detection
try:
    import face_recognition
    USING_FR_LIB = True
    logger.info("Using face_recognition library")
except ImportError:
    USING_FR_LIB = False
    logger.info("face_recognition library not available, using OpenCV's face detection")
    # Load OpenCV's pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def encode_face(image_rgb):
    """
    Check if a face is detectable in the image
    
    Args:
        image_rgb: RGB image array
        
    Returns:
        True if face is detected, False otherwise
    """
    if image_rgb is None:
        logger.error("Invalid image provided for encoding")
        return False
        
    if USING_FR_LIB:
        try:
            # Find all face locations in the image
            face_locations = face_recognition.face_locations(image_rgb, model="hog")
            
            if not face_locations:
                logger.warning("No faces detected in the image")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error detecting face with face_recognition: {str(e)}")
            return False
    else:
        # Fallback to OpenCV
        try:
            # Convert RGB to grayscale for OpenCV face detection
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            
            # Enhance image contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            
            # Detect faces with improved parameters
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,  # Lower value for better detection at different sizes
                minNeighbors=4,    # Lower slightly to detect more faces
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            if len(faces) == 0:
                logger.warning("No faces detected in the image using OpenCV")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error detecting face with OpenCV: {str(e)}")
            return False

def get_face_encoding_from_image_path(image_path):
    """
    Get face encoding from an image file
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Face encoding or None if no face detected
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image from {image_path}")
            return None
            
        # Apply image enhancement for better face detection
        image = enhance_image_for_face_detection(image)
            
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if USING_FR_LIB:
            # Try both HOG and CNN models if faces not detected with HOG
            face_locations = face_recognition.face_locations(image_rgb, model="hog")
            
            if not face_locations:
                logger.info(f"Attempting CNN model as HOG didn't detect faces in: {image_path}")
                try:
                    face_locations = face_recognition.face_locations(image_rgb, model="cnn")
                except Exception:
                    logger.warning("CNN model not available or failed, continuing with HOG results")
            
            if not face_locations:
                logger.warning(f"No faces detected in image: {image_path}")
                return None
                
            # Generate encoding for the first face
            face_encodings = face_recognition.face_encodings(image_rgb, [face_locations[0]])
            
            if not face_encodings:
                logger.warning(f"Face found but couldn't be encoded in: {image_path}")
                return None
                
            return face_encodings[0]
        else:
            # Fallback to OpenCV with enhanced detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            
            # Try multiple face detection cascades for better results
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=4,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # If no faces detected, try with LBP cascade as fallback
            if len(faces) == 0:
                try:
                    lbp_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'lbpcascade_frontalface_improved.xml')
                    if not lbp_cascade.empty():
                        faces = lbp_cascade.detectMultiScale(
                            gray,
                            scaleFactor=1.1,
                            minNeighbors=4,
                            minSize=(30, 30),
                            flags=cv2.CASCADE_SCALE_IMAGE
                        )
                except Exception as e:
                    logger.warning(f"LBP cascade fallback failed: {str(e)}")
            
            if len(faces) == 0:
                logger.warning(f"No faces detected in image using OpenCV: {image_path}")
                return None
                
            # Get the largest face
            if len(faces) > 1:
                # If multiple faces, use the largest one
                largest_area = 0
                largest_face_idx = 0
                
                for i, (x, y, w, h) in enumerate(faces):
                    if w*h > largest_area:
                        largest_area = w*h
                        largest_face_idx = i
                        
                (x, y, w, h) = faces[largest_face_idx]
            else:
                (x, y, w, h) = faces[0]
                
            # Extract face ROI and resize to standard size
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (128, 128))
            
            # Apply histogram equalization to improve feature extraction
            face_roi = clahe.apply(face_roi)
            
            # Normalize pixel values
            face_roi = face_roi.astype(np.float32) / 255.0
            
            # Flatten the array as a simple encoding
            return face_roi.flatten()
            
    except Exception as e:
        logger.error(f"Error getting face encoding from image: {str(e)}")
        return None

def recognize_face(image_rgb, students, tolerance=0.6):
    """
    Recognize faces in an image using stored face images
    
    Args:
        image_rgb: RGB image array
        students: List of Student objects
        tolerance: Recognition tolerance (lower = stricter)
        
    Returns:
        Tuple of (face_locations, face_names)
    """
    if image_rgb is None:
        logger.error("Invalid image provided for recognition")
        return [], []
        
    face_locations = []
    face_names = []
    
    if not students:
        logger.warning("No students provided")
        return [], []
    
    # Cache for face encodings to improve performance
    encoding_cache_file = os.path.join(settings.MEDIA_ROOT, 'encoding_cache.pkl')
    cache_outdated = True
    known_encodings = []
    known_names = []
    encoding_metadata = {}
    
    # Check if cache exists and is still valid
    if os.path.exists(encoding_cache_file):
        try:
            with open(encoding_cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                encoding_metadata = cached_data.get('metadata', {})
                
                # Check if cache is still valid
                cache_outdated = False
                for student in students:
                    if student.face_image and student.face_image.name:
                        image_path = os.path.join(settings.MEDIA_ROOT, student.face_image.name)
                        if image_path not in encoding_metadata:
                            cache_outdated = True
                            break
                        elif os.path.exists(image_path):
                            mtime = os.path.getmtime(image_path)
                            if mtime > encoding_metadata.get(image_path, {}).get('mtime', 0):
                                cache_outdated = True
                                break
                
                # If cache is valid, use it
                if not cache_outdated:
                    known_encodings = cached_data.get('encodings', [])
                    known_names = cached_data.get('names', [])
                    logger.info("Using cached face encodings")
        except Exception as e:
            logger.warning(f"Error reading encoding cache: {str(e)}")
            cache_outdated = True
    
    # If cache is outdated or doesn't exist, regenerate encodings
    if cache_outdated:
        logger.info("Regenerating face encodings cache")
        known_encodings = []
        known_names = []
        encoding_metadata = {}
        
        for student in students:
            if student.face_image and student.face_image.name:
                # Get the full path to the image
                image_path = os.path.join(settings.MEDIA_ROOT, student.face_image.name)
                
                if os.path.exists(image_path):
                    encoding = get_face_encoding_from_image_path(image_path)
                    if encoding is not None:
                        known_encodings.append(encoding)
                        known_names.append(student.student_id)
                        
                        # Save metadata for cache validation
                        encoding_metadata[image_path] = {
                            'mtime': os.path.getmtime(image_path),
                            'size': os.path.getsize(image_path)
                        }
                else:
                    logger.warning(f"Face image not found for student {student.student_id}: {image_path}")
        
        # Save cache for future use
        try:
            cache_data = {
                'encodings': known_encodings,
                'names': known_names,
                'metadata': encoding_metadata,
                'timestamp': time.time()
            }
            
            os.makedirs(os.path.dirname(encoding_cache_file), exist_ok=True)
            with open(encoding_cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info("Face encodings cache saved successfully")
        except Exception as e:
            logger.warning(f"Error saving encoding cache: {str(e)}")
    
    if not known_encodings:
        logger.warning("No valid face encodings found")
        return [], []
    
    # Try different image enhancements to improve recognition
    enhanced_img_rgb = enhance_image_for_face_detection(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    enhanced_img_rgb = cv2.cvtColor(enhanced_img_rgb, cv2.COLOR_BGR2RGB)
    
    if USING_FR_LIB:
        try:
            # Find all face locations in the image
            face_locations = face_recognition.face_locations(image_rgb, model="hog")
            
            if not face_locations:
                # Try with enhanced image
                face_locations = face_recognition.face_locations(enhanced_img_rgb, model="hog")
                
                if not face_locations:
                    logger.warning("No faces detected in image for recognition")
                    return [], []
                else:
                    # Use enhanced image for encoding if it worked better
                    image_rgb = enhanced_img_rgb
                
            # Get encodings for all faces in the image
            face_encodings = face_recognition.face_encodings(image_rgb, face_locations)
            
            # Match each face to our known faces
            for face_encoding in face_encodings:
                name = "Unknown"
                
                if known_encodings:
                    # Compare face with all known faces
                    matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=tolerance)
                    
                    # Use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                    
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        best_distance = face_distances[best_match_index]
                        
                        # Add confidence metric - lower distance = higher confidence
                        confidence = 1 - best_distance
                        
                        # Only accept match if confidence is high enough
                        if matches[best_match_index] and confidence >= 0.5:
                            name = known_names[best_match_index]
                            logger.info(f"Face matched with confidence: {confidence:.2f}")
                
                face_names.append(name)
                
        except Exception as e:
            logger.error(f"Error recognizing faces with face_recognition: {str(e)}")
            return [], []
    else:
        # Fallback to OpenCV
        try:
            # Convert RGB to grayscale for OpenCV face detection
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            
            # Detect faces with improved parameters
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=4,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            for (x, y, w, h) in faces:
                # Extract face ROI and resize
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (128, 128))
                
                # Apply histogram equalization
                face_roi = clahe.apply(face_roi)
                
                # Normalize pixel values
                face_roi = face_roi.astype(np.float32) / 255.0
                
                # Flatten to match our simple encoding format
                encoding = face_roi.flatten()
                
                # Convert face location format to match face_recognition library
                face_locations.append((y, x+w, y+h, x))
                
                # Match against known encodings
                name = "Unknown"
                min_distance = float('inf')
                
                for i, enc in enumerate(known_encodings):
                    if enc is not None:
                        # For OpenCV method, use multiple distance metrics and combine
                        euclidean_distance = np.linalg.norm(enc - encoding)
                        cosine_similarity = np.dot(enc, encoding) / (np.linalg.norm(enc) * np.linalg.norm(encoding))
                        
                        # Convert cosine similarity to distance (1 - similarity)
                        cosine_distance = 1 - cosine_similarity
                        
                        # Combined distance metric (weighted average)
                        distance = 0.7 * euclidean_distance + 0.3 * cosine_distance
                        
                        # Threshold for matching (lower is better)
                        if distance < min_distance and distance < 0.4:  # Stricter threshold
                            min_distance = distance
                            name = known_names[i]
                
                face_names.append(name)
                
        except Exception as e:
            logger.error(f"Error recognizing faces with OpenCV: {str(e)}")
            return [], []
    
    return face_locations, face_names

def enhance_image_for_face_detection(image):
    """
    Enhance image quality for better face detection
    
    Args:
        image: OpenCV image in BGR format
    
    Returns:
        Enhanced image
    """
    try:
        # Check if image is already grayscale
        if len(image.shape) == 2 or image.shape[2] == 1:
            gray = image.copy()
            if len(image.shape) == 3:
                gray = gray.reshape(gray.shape[0], gray.shape[1])
        else:
            # Convert BGR to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply noise reduction
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Process LAB channels for color images
        if len(image.shape) == 3 and image.shape[2] == 3:
            lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            
            # Sharpen the image
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            enhanced_img = cv2.filter2D(enhanced_img, -1, kernel)
            
            return enhanced_img
        else:
            # For grayscale images
            enhanced_img = clahe.apply(gray)
            return enhanced_img
    except Exception as e:
        logger.error(f"Error enhancing image: {str(e)}")
        return image  # Return original image in case of error

def check_image_quality(image):
    """
    Check image quality for face detection
    
    Args:
        image: OpenCV image
    
    Returns:
        Dictionary with quality metrics and issues list
    """
    issues = []
    
    # Check image dimensions
    height, width = image.shape[:2]
    if width < 200 or height < 200:
        issues.append("Image resolution too low")
    
    # Check brightness
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    brightness = np.mean(gray)
    if brightness < 40:
        issues.append("Image too dark")
    elif brightness > 220:
        issues.append("Image too bright")
    
    # Check contrast
    contrast = np.std(gray)
    if contrast < 20:
        issues.append("Image contrast too low")
    
    # Check blur
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 100:
        issues.append("Image is blurry")
    
    return {
        "brightness": brightness,
        "contrast": contrast, 
        "sharpness": laplacian_var,
        "issues": issues
    }

def preprocess_student_images():
    """
    Preprocess and improve quality of stored student face images
    
    Returns:
        Number of images processed
    """
    try:
        faces_dir = os.path.join(settings.MEDIA_ROOT, 'faces')
        if not os.path.exists(faces_dir):
            logger.warning(f"Faces directory does not exist: {faces_dir}")
            return 0
            
        image_count = 0
        processed_dir = os.path.join(faces_dir, 'processed')
        os.makedirs(processed_dir, exist_ok=True)
        
        for filename in os.listdir(faces_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(faces_dir, filename)
                
                # Skip directories and already processed files
                if os.path.isdir(file_path) or 'processed_' in filename:
                    continue
                    
                try:
                    # Load and enhance image
                    image = cv2.imread(file_path)
                    if image is None:
                        logger.warning(f"Could not read image: {file_path}")
                        continue
                        
                    # Check quality
                    quality = check_image_quality(image)
                    
                    # Only process if there are issues
                    if quality['issues']:
                        logger.info(f"Processing {filename}, issues: {quality['issues']}")
                        
                        # Enhance image
                        enhanced = enhance_image_for_face_detection(image)
                        
                        # Save processed image
                        processed_path = os.path.join(processed_dir, f"processed_{filename}")
                        cv2.imwrite(processed_path, enhanced)
                        
                        image_count += 1
                except Exception as e:
                    logger.error(f"Error processing image {filename}: {str(e)}")
                    
        return image_count
    except Exception as e:
        logger.error(f"Error preprocessing student images: {str(e)}")
        return 0

# Import time here to avoid circular import
import time