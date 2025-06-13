from fastapi import APIRouter, UploadFile, File 
import numpy as np
import cv2

from backend.computer_vision.object_detector import ObjectDetector
from backend.computer_vision.feature_tracker import FeatureTracker
from backend.computer_vision.utils import decode_base64_to_image, encode_image_to_base64

router = APIRouter(prefix="/vision", tags=["vision"])

@router.post("/detect")
async def detect_objects(image: UploadFile = File(...)):
    data = await image.read()
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    det = ObjectDetector().detect(frame)
    return {"detections": det}

@router.post("/track")
async def track_features(image: UploadFile = File(...)):
    data = await image.read()
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    pts = FeatureTracker().track(frame)
    return {"keypoints": [kp.pt for kp in pts]}

@router.get("/health")
async def health_check():
    return {"vision_service": "ok"}