import torch
import cv2
import numpy as np

class ObjectDetector:
    """
    YOLOv5-based object detector (via torch.hub).
    """
    def __init__(self, model_name: str="yolov5s"):
        self.model = torch.hub.load("ultralytics/yolov5", model_name, pretrained=True)
        self.model.eval()

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run inference on BGR numpy image.
        Returns a list of {label, confidence, bbox:[x1,y1,x2,y2]}.
        """
        results = self.model(frame[..., ::-1])  # BGR→RGB
        detections = []
        for *xyxy, conf, cls in results.xyxy[0].tolist():
            detections.append({
               "bbox": [*xyxy],
               "confidence": conf,
               "label": self.model.names[int(cls)]
            })
        return detections