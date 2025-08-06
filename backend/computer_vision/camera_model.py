import cv2
import numpy as np

class CameraModel:
    """
    Simple pin-hole camera model with optional distortion.
    """
    def __init__(self, intrinsics: np.ndarray, dist_coeffs: np.ndarray = None):
        self.K = intrinsics        # 3×3 camera matrix
        self.dist = dist_coeffs     # distortion coefficients

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        if self.dist is None:
            return frame
        return cv2.undistort(frame, self.K, self.dist)

    def project_points(self, pts_3d: np.ndarray) -> np.ndarray:
        """Project Nx3 world points into image plane (N×2)."""
        pts, _ = cv2.projectPoints(
            pts_3d.reshape(-1,1,3),
            rvec=np.zeros(3), tvec=np.zeros(3),
            cameraMatrix=self.K, distCoeffs=self.dist
        )
        return pts.reshape(-1,2)

    def capture_frame(self):
        """Placeholder – override or hook into your Unity feed."""
        raise NotImplementedError("Implement frame capture from your engine")