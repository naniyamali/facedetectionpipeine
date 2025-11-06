# Face Verification Tool

A reliable face verification tool using PyTorch-based face detection and recognition. This implementation uses MTCNN for face detection and InceptionResnetV1 (pretrained on VGGFace2) for face recognition.

## Features

- Robust face detection using MTCNN
- State-of-the-art face embeddings using InceptionResnetV1
- Easy-to-use interface that returns "matched"/"not matched"
- Detailed confidence scores and proper error handling
- Works reliably on Windows (no TensorFlow dependency issues)

## Installation

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install facenet-pytorch opencv-python-headless
```

## Usage

### As a module

```python
from deepface_tool.verify_pan_wrapper import verify_pan

result = verify_pan(
    "path/to/image1.jpg",
    "path/to/image2.jpg"
)
print(result)  # "matched" or "not matched"
```

### Direct script usage

```python
python verify_with_facenet.py  # Uses default test images
```

## API

The main verification function returns three values:
- `is_match`: Boolean indicating if faces match
- `confidence`: Similarity score (0-1, higher is more similar)
- `success`: Boolean indicating if face detection succeeded

## Requirements

- Python 3.8+
- PyTorch
- facenet-pytorch
- opencv-python-headless
- PIL