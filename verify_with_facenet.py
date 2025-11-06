"""
Face verification using facenet-pytorch's MTCNN for detection and InceptionResnetV1 for embeddings.
More reliable on Windows than DeepFace's TensorFlow dependency chain.
"""
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import numpy as np
from PIL import Image
import os

import cv2
import numpy as np

def preprocess_face(image, target_size=(112, 112)):
    """
    Preprocess an input image to detect the largest face, align, resize, and normalize it.
    first preprocessign inorder to extract face from image by using cascade classifier.
    Args:
    - image: Input image as a NumPy array (BGR format, as loaded by cv2).
    - target_size: Desired output size (width, height) for the face image.
     accordingly we could change the target size
    
    Returns:
    - preprocessed_face: Face image after detection, alignment, resizing, and normalization.
                         Returns None if no face is detected.
    """
    # Load pre-trained OpenCV frontal face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert to grayscale for detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    
    if len(faces) == 0:
        return None  # No face detected
    
    # Select the largest detected face (assumed primary)
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    
    # Crop the face region from original image
    face_img = image[y:y+h, x:x+w]
    
    # Optional: Align face based on eyes or landmarks - skipped here, more complex
    
    # Resize face to target size
    face_resized = cv2.resize(face_img, target_size)
    
    # Normalize pixel values to [0,1]
    face_normalized = face_resized.astype(np.float32) / 255.0
    
    # Convert to RGB if needed (from BGR as OpenCV loads)
    face_rgb = cv2.cvtColor(face_normalized, cv2.COLOR_BGR2RGB)
    
    return face_rgb


def load_image(image_path):
    """Load and preprocess an image for face detection."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Read with cv2 and convert BGR to RGB
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb, Image.fromarray(img_rgb)


def verify_faces(img1_path, img2_path, min_face_size=20, threshold=0.6):
    """
    Verify if two face images show the same person.
    Returns (is_match, confidence_score, detection_success)
    """
    device = torch.device('cpu')
    
    # Initialize the face detector and recognition model
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=min_face_size,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device, keep_all=False
    )
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    try:
        # Load and detect faces in both images
        img1_rgb, img1_pil = load_image(img1_path)
        img2_rgb, img2_pil = load_image(img2_path)
        
        print(f"Loaded images - shapes: {img1_rgb.shape}, {img2_rgb.shape}")
        
        # Get face tensors (returns None if no face detected)
        face1 = mtcnn(img1_pil)
        face2 = mtcnn(img2_pil)
        
        if face1 is None or face2 is None:
            print("Warning: Could not detect face in one or both images")
            return False, 0.0, False
        
        # Generate embeddings
        with torch.no_grad():
            embedding1 = resnet(face1.unsqueeze(0))
            embedding2 = resnet(face2.unsqueeze(0))
        
        # Calculate cosine similarity
        cos = torch.nn.CosineSimilarity(dim=1)
        similarity = cos(embedding1, embedding2).item()
        
        # Convert similarity score to distance (lower is more similar)
        distance = 1.0 - similarity
        is_match = distance < threshold
        
        print(f"Face similarity distance: {distance:.4f} (threshold: {threshold})")
        return is_match, similarity, True
        
    except Exception as e:
        print(f"Error during verification: {str(e)}")
        return False, 0.0, False


def main():
    # Default paths (same as original script)
    img1_path = r"C:\Users\Y NANI\OneDrive\Pictures\Camera Roll\WIN_20251106_11_01_34_Pro.jpg"
    img2_path = r"C:\Users\Y NANI\Downloads\images\mediacnani.jpg"
    print("Starting face verification...")
    img1=preprocess_face(cv2.imread(img1_path))
    img2=preprocess_face(cv2.imread(img2_path))
    is_match, confidence, success = verify_faces(img1, img2)
    
    if success:
        result = "matched" if is_match else "not matched"
        print(f"\nFinal verification result: {result} (confidence: {confidence:.4f})")
    else:
        print("\nVerification failed - could not detect faces properly")


if __name__ == '__main__':
    main()