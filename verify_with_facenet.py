

import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import numpy as np
from PIL import Image
import os


#we are passing or input images here the face detection goes under haarcasade the face extraction
def preprocess_face(image, target_size=(160, 160)):
    """
    Detect and crop the largest face in an image using Haar Cascade.
    Returns a normalized RGB NumPy array (float32).
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) == 0:
        return None

    # Select the largest detected face (primary face) bring out bigggest rectangle
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    face_img = image[y:y + h, x:x + w]

    # Resize to match ResNet input (160x160)
    face_resized = cv2.resize(face_img, target_size)

    # Normalize to [0,1]
    face_normalized = face_resized.astype(np.float32) / 255.0

    # Convert BGR → RGB
    face_rgb = cv2.cvtColor(face_normalized, cv2.COLOR_BGR2RGB)
    return face_rgb


#using cv2.imread()
def load_image(image_path):
    """Loads an image from a path and returns both np.array and PIL.Image."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #here opencvnumpymatplotlib expects ndin array , they both are same but keeping them in pixel wrapper it returns under pixel object  while passing mtcnn expects pixel so we are doing like this
    return img_rgb, Image.fromarray(img_rgb)


# -------------------------------------------------------------
# 3. Hybrid Face Verification Function
# -------------------------------------------------------------
def verify_faces(img1_path, img2_path, threshold=0.6):
    """
    a sub method here written is extract face

    Returns:
        True if the face is detected, False otherwise.
        (is_match: bool, confidence: float, success: bool)
    """

    device = torch.device('cpu')

    # Load detection & recognition models
    mtcnn = MTCNN(image_size=160, device=device, keep_all=False)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    def extract_face(img_path):

        img_rgb, img_pil = load_image(img_path)
        #print(img_rgb)
        #print("jsd",img_pil)

        # multi task cascade network c
        face_tensor = mtcnn(img_pil)
        if face_tensor is not None:
            print(f" MTCNN successfully detected a face in {os.path.basename(img_path)}")
            return face_tensor.unsqueeze(0)

        # --- Fallback to Haar Cascade --- if mtcnn fails to detect then go it will go haarcascade
        print(f" MTCNN failed for {os.path.basename(img_path)} → trying Haar Cascade...")
        img_cv = cv2.imread(img_path)
        face_np = preprocess_face(img_cv)
        if face_np is None:
            print(f" Haar Cascade also failed for {os.path.basename(img_path)}")
            return None

        # Convert NumPy → Torch Tensor (C,H,W)
        face_tensor = torch.tensor(face_np).permute(2, 0, 1).unsqueeze(0).float()
        return face_tensor

    # Extract faces
    face1 = extract_face(img1_path)
    face2 = extract_face(img2_path)

    if face1 is None or face2 is None:
        print("Could not detect face in one or both images.")
        return False, 0.0, False

    # embeddings will be passed in resnet i mean the faces here
    with torch.no_grad():
        emb1 = resnet(face1)
        emb2 = resnet(face2)
        #print("embi",emb1)
        #print("emb2",emb2)

    # --- Cosine Similarity ---,dotproduct of vectors by root of sum of squares of distances
    cos = torch.nn.CosineSimilarity(dim=1)
    similarity = cos(emb1, emb2).item()
    distance = 1 - similarity
    is_match = distance < threshold

    print("\n--- Verification Results ---")
    print(f"Cosine similarity : {similarity:.4f}")
    print(f"Distance           : {distance:.4f}")
    print(f"Threshold          : {threshold:.2f}")
    print(f"Result             : {'MATCH ' if is_match else 'NOT MATCH '}")

    return is_match, similarity, True



def main():
    # Replace with your image paths
    #for testing purpose we could go as low similarity we could be proceed
    img1_paths = r"C:\Users\Y NANI\OneDrive\Pictures\Camera Roll\WIN_20251107_12_35_44_Pro.jpg"
    img2_paths = [r"C:\Users\Y NANI\Downloads\images\WIN_20251105_14_36_40_Pro.jpg",r"C:\Users\Y NANI\OneDrive\Pictures\Camera Roll\WIN_20251106_11_01_34_Pro.jpg",r"C:\Users\Y NANI\OneDrive\Pictures\Camera Roll\WIN_20251105_14_12_04_Pro.jpg"]

    print(" Starting face verification...\n")
    for i in img2_paths,img1_paths:
        is_match, confidence, success = verify_faces(img1_path, i)
        if success:
            result = "matched" if is_match else "not matched"
            print(f"\n Final Result: Faces are {result} (confidence: {confidence:.4f})")
        else:
            print("\n Verification failed - no valid faces detected.")


#if we launch the process here the flow will go with verify faces then the extra face will be called
if __name__ == '__main__':
    main()
