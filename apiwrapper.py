from verify_with_facenet import verify_faces
import cv2
import matplotlib.pyplot as plt
from flask import Flask, jsonify, render_template_string

app = Flask(__name__)


# Add CORS headers to responses so the frontend can call the API even when
# served from another origin (or opened as a local file). This keeps changes
# minimal and avoids adding a new dependency.
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return response

# HTML template as a string (embedded for simplicity—no separate file needed)
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Face Detection</title>
</head>
<body>
    <h1>Face Verification Tool</h1>
    <button onclick="getFaceDetection()">Get Face Detection</button>

    <script>
        async function getFaceDetection() {
            try {
                let response = await fetch('/api/returnresult', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({})
                });
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                let res = await response.json();
                alert(res.msg);
            } catch (error) {
                alert('Error: ' + error.message);
                console.error('Fetch error:', error);
            }
        }
    </script>
</body>
</html>
'''


@app.route('/', methods=['GET'])
def index():
    """Serve the HTML page at the root URL."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/returnresult', methods=['POST'])
def verify_pan():
    # Hardcoded paths (replace with dynamic if uploading later)
    img1_path = r"C:\Users\Y NANI\OneDrive\Pictures\Camera Roll\WIN_20251107_12_35_44_Pro.jpg"
    img2_path = r"C:\Users\Y NANI\Downloads\images\WIN_20251105_14_36_40_Pro.jpg"

    # Wrap the verification in a try/except so failures are returned as JSON
    # (helps debugging when the route is hit over HTTP). Keep response
    # structure minimal — success returns {'msg': 'matched'|'not matched'}.
    try:
        is_match, confidence, success = verify_faces(img1_path, img2_path)
        if not success:
            return jsonify({'msg': 'not matched'})

        result = 'matched' if is_match else 'not matched'
        return jsonify({'msg': result})
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        # Return error details to caller (development helper). If you prefer
        # not to expose tracebacks, remove `traceback` from the response.
        return jsonify({'error': str(e), 'traceback': tb}), 500


if __name__ == '__main__':
    app.run(debug=True)