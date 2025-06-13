import cv2
from typing import List

class FeatureTracker:
    """
    Detect & track keypoints using ORB + brute‐force matcher.
    """
    def __init__(self, max_features:int=500):
        self.detector = cv2.ORB_create(nfeatures=max_features)
        self.matcher  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_kp  = None
        self.prev_des = None

    def detect(self, frame: np.ndarray):
        kp, des = self.detector.detectAndCompute(frame, None)
        self.prev_kp, self.prev_des = kp, des
        return kp

    def track(self, frame: np.ndarray) -> List[cv2.KeyPoint]:
        kp, des = self.detector.detectAndCompute(frame, None)
        if self.prev_des is None:
            self.prev_kp, self.prev_des = kp, des
            return []
        matches = self.matcher.match(self.prev_des, des)
        matched_kp = [kp[m.trainIdx] for m in matches]
        self.prev_kp, self.prev_des = kp, des
        return matched_kp