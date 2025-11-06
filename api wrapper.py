from verify_with_facenet import verify_faces
import Flask
@app.route('/api/returnresult', methods=['POST'])
def verify_pan(img1_path, img2_path):
    """Drop-in replacement for the original verify_pan function.
    Returns "matched" or "not matched" to maintain compatibility."""
    is_match, confidence, success = verify_faces(img1_path, img2_path)
    if not success:
        return "not matched"  # Same behavior as original on error
    return "matched" if is_match else "not matched"