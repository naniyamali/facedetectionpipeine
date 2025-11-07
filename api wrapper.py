from verify_with_facenet import verify_faces
import cv2
import matplotlib.pyplot as plt
#import Flask
"""
@app.route('/api/returnresult', methods=['POST'])
def verify_pan(img1_path, img2_path):
    
    is_match, confidence, success = verify_faces(img1_path, img2_path)
    if not success:
        return "not matched"  # Same behavior as original on error
    return "matched" if is_match else "not matched" """

import cv2

img1_path = r"C:\Users\Y NANI\Downloads\images\WhatsApp Image 2025-11-07 at 11.34.51_49fb61a0.jpg"

img=cv2.imread(img1_path)
plt.imshow(img)

plt.show()
#cv2.imshow("image", img)
#cv2.waitKey(0)