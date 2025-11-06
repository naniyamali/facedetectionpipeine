"""Simple runner for DeepFace.verify with better diagnostics.
Place images at the hard-coded paths or pass two paths as command-line arguments.
"""
from deepface import DeepFace
import cv2
import os
import sys
import traceback


def verify_images(img1_path, img2_path, detector='retinaface'):
    # Validate paths
    if not os.path.isfile(img1_path):
        raise FileNotFoundError(f"Image not found: {img1_path}")
    if not os.path.isfile(img2_path):
        raise FileNotFoundError(f"Image not found: {img2_path}")

    # Quick sanity: ensure images can be read
    i1 = cv2.imread(img1_path)
    i2 = cv2.imread(img2_path)
    print("read shapes:", getattr(i1, 'shape', None), getattr(i2, 'shape', None))
    if i1 is None or i2 is None:
        raise RuntimeError("cv2 failed to read one of the images")

    # Print a small sample of pixel values for extra diagnostics
    try:
        print('i1 sample pixel [0,0]:', i1[0,0])
        print('i2 sample pixel [0,0]:', i2[0,0])
    except Exception:
        pass

    # Try verification
    try:
        print(f"Calling DeepFace.verify with detector_backend='{detector}'\n")
        res = DeepFace.verify(img1_path, img2_path, detector_backend=detector)
        print("Result:", res)
        return res
    except Exception as e:
        print(f"Verification with detector '{detector}' threw exception:\n", repr(e))
        # If retinaface fails, try a fallback detector
        if detector != 'mtcnn':
            print("Falling back to detector_backend='mtcnn' and retrying...\n")
            return verify_images(img1_path, img2_path, detector='mtcnn')
        else:
            raise


if __name__ == '__main__':
    # allow passing image paths as args
    if len(sys.argv) >= 3:
        img1 = sys.argv[1]
        img2 = sys.argv[2]
    else:
        img1 = r"C:\Users\Y NANI\Downloads\images\WIN_20251105_14_36_40_Pro.jpg"
        img2 = r"C:\Users\Y NANI\Downloads\images\captured_image.jpg"

    try:
        result = verify_images(img1, img2)
        print('\nFinal verification: matched' if result.get('verified') else '\nFinal verification: not matched')
    except Exception as exc:
        print('\nScript failed with exception:')
        traceback.print_exc()
