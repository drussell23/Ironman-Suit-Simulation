import base64
import cv2
import numpy as np

def encode_image_to_base64(frame: np.ndarray) -> str:
    _, buf = cv2.imencode('.jpg', frame)
    return base64.b64encode(buf).decode('utf-8')

def decode_base64_to_image(data: str) -> np.ndarray:
    arr = np.frombuffer(base64.b64decode(data), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)