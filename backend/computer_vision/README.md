# Computer Vision Package

Provides camera modeling, feature tracking, object detection and
a lightweight stream client for integrating Unity render feeds.

Modules:
- `camera_model.py`    – pin‐hole + distortion, project/undistort.
- `feature_tracker.py` – ORB + BFMatcher for keypoint tracking.
- `object_detector.py` – YOLOv5 detector via torch.hub.
- `stream_client.py`   – UDP JPEG frame receiver.
- `utils.py`           – base64↔image helpers.

Usage:
```python
from backend.computer_vision import CameraModel, FeatureTracker, ObjectDetector, StreamClient