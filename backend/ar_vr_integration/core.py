"""
Core AR/VR framework and utilities for Iron Man suit.

This module provides the foundational components for AR/VR integration,
including coordinate systems, rendering pipelines, device management,
and shared utilities.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import threading
import time
import logging
from abc import ABC, abstractmethod


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Supported AR/VR device types"""
    HMD_VR = "hmd_vr"  # VR headset
    HMD_AR = "hmd_ar"  # AR headset
    HOLOGRAPHIC = "holographic"  # Holographic display
    PROJECTOR = "projector"  # Spatial projector
    MOBILE_AR = "mobile_ar"  # Mobile AR device


class RenderMode(Enum):
    """Rendering modes"""
    STEREO_VR = "stereo_vr"
    MONO_AR = "mono_ar"
    HOLOGRAPHIC_3D = "holographic_3d"
    PROJECTION_MAPPING = "projection_mapping"


class CoordinateSpace(Enum):
    """Coordinate space definitions"""
    WORLD = "world"  # Real world coordinates
    DEVICE = "device"  # Device-relative coordinates
    USER = "user"  # User-centric coordinates
    SCREEN = "screen"  # Screen/display coordinates


@dataclass
class Transform:
    """3D transformation data"""
    position: np.ndarray = None
    rotation: np.ndarray = None  # Quaternion
    scale: np.ndarray = None
    
    def __post_init__(self):
        if self.position is None:
            self.position = np.zeros(3)
        if self.rotation is None:
            self.rotation = np.array([0, 0, 0, 1])  # Identity quaternion
        if self.scale is None:
            self.scale = np.ones(3)
    
    @property
    def matrix(self) -> np.ndarray:
        """Get 4x4 transformation matrix"""
        mat = np.eye(4)
        
        # Apply scale
        mat[:3, :3] = np.diag(self.scale)
        
        # Apply rotation (quaternion to matrix)
        q = self.rotation
        mat[:3, :3] = mat[:3, :3] @ np.array([
            [1-2*(q[1]**2+q[2]**2), 2*(q[0]*q[1]-q[2]*q[3]), 2*(q[0]*q[2]+q[1]*q[3])],
            [2*(q[0]*q[1]+q[2]*q[3]), 1-2*(q[0]**2+q[2]**2), 2*(q[1]*q[2]-q[0]*q[3])],
            [2*(q[0]*q[2]-q[1]*q[3]), 2*(q[1]*q[2]+q[0]*q[3]), 1-2*(q[0]**2+q[1]**2)]
        ])
        
        # Apply translation
        mat[:3, 3] = self.position
        
        return mat


@dataclass
class RenderTarget:
    """Render target configuration"""
    width: int
    height: int
    fov: float  # Field of view in degrees
    near_plane: float = 0.1
    far_plane: float = 1000.0
    is_stereo: bool = False
    eye_separation: float = 0.063  # meters


class ARVRDevice(ABC):
    """Abstract base class for AR/VR devices"""
    
    def __init__(self, device_id: str, device_type: DeviceType):
        self.device_id = device_id
        self.device_type = device_type
        self.is_connected = False
        self.transform = Transform()
        self.render_target = None
        
    @abstractmethod
    def connect(self) -> bool:
        """Connect to the device"""
        pass
    
    @abstractmethod
    def disconnect(self):
        """Disconnect from the device"""
        pass
    
    @abstractmethod
    def get_tracking_data(self) -> Dict[str, Any]:
        """Get current tracking data from device"""
        pass
    
    @abstractmethod
    def render(self, frame_data: np.ndarray):
        """Render frame to device"""
        pass


class ARVRCore:
    """
    Core AR/VR system manager.
    
    Handles device management, coordinate transformations,
    rendering pipeline, and system-wide configuration.
    """
    
    def __init__(self):
        self.devices: Dict[str, ARVRDevice] = {}
        self.active_device: Optional[str] = None
        self.render_mode = RenderMode.MONO_AR
        
        # Coordinate system management
        self.coordinate_transforms: Dict[str, Transform] = {
            CoordinateSpace.WORLD.value: Transform(),
            CoordinateSpace.DEVICE.value: Transform(),
            CoordinateSpace.USER.value: Transform(),
            CoordinateSpace.SCREEN.value: Transform()
        }
        
        # Rendering pipeline
        self.render_pipeline = []
        self.frame_buffer = None
        self.render_resolution = (1920, 1080)
        
        # System state
        self.is_running = False
        self.render_thread = None
        self.tracking_thread = None
        self.fps_target = 90  # Target FPS for VR
        self.last_frame_time = time.time()
        
        # Performance metrics
        self.metrics = {
            'fps': 0,
            'frame_time': 0,
            'tracking_latency': 0,
            'render_latency': 0,
            'total_latency': 0
        }
        
        # Configuration
        self.config = {
            'enable_reprojection': True,
            'enable_foveated_rendering': False,
            'enable_motion_smoothing': True,
            'tracking_prediction_ms': 20,
            'vsync': True
        }
        
        # Callbacks
        self.callbacks = {
            'on_device_connected': None,
            'on_device_disconnected': None,
            'on_tracking_lost': None,
            'on_tracking_recovered': None
        }
        
        self._initialize_render_pipeline()
    
    def _initialize_render_pipeline(self):
        """Initialize the rendering pipeline stages"""
        self.render_pipeline = [
            self._stage_tracking_prediction,
            self._stage_culling,
            self._stage_rendering,
            self._stage_post_processing,
            self._stage_reprojection,
            self._stage_output
        ]
    
    def add_device(self, device: ARVRDevice) -> bool:
        """Add an AR/VR device to the system"""
        if device.device_id in self.devices:
            logger.warning(f"Device {device.device_id} already exists")
            return False
        
        self.devices[device.device_id] = device
        
        if device.connect():
            device.is_connected = True
            if not self.active_device:
                self.active_device = device.device_id
            
            if self.callbacks['on_device_connected']:
                self.callbacks['on_device_connected'](device)
            
            return True
        
        return False
    
    def remove_device(self, device_id: str):
        """Remove a device from the system"""
        if device_id in self.devices:
            device = self.devices[device_id]
            device.disconnect()
            del self.devices[device_id]
            
            if self.active_device == device_id:
                self.active_device = None
                # Switch to next available device
                if self.devices:
                    self.active_device = list(self.devices.keys())[0]
            
            if self.callbacks['on_device_disconnected']:
                self.callbacks['on_device_disconnected'](device)
    
    def set_active_device(self, device_id: str) -> bool:
        """Set the active rendering device"""
        if device_id in self.devices and self.devices[device_id].is_connected:
            self.active_device = device_id
            self._update_render_mode()
            return True
        return False
    
    def _update_render_mode(self):
        """Update render mode based on active device"""
        if self.active_device:
            device = self.devices[self.active_device]
            
            if device.device_type == DeviceType.HMD_VR:
                self.render_mode = RenderMode.STEREO_VR
            elif device.device_type == DeviceType.HMD_AR:
                self.render_mode = RenderMode.MONO_AR
            elif device.device_type == DeviceType.HOLOGRAPHIC:
                self.render_mode = RenderMode.HOLOGRAPHIC_3D
            elif device.device_type == DeviceType.PROJECTOR:
                self.render_mode = RenderMode.PROJECTION_MAPPING
    
    def start(self):
        """Start the AR/VR system"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start tracking thread
        self.tracking_thread = threading.Thread(target=self._tracking_loop)
        self.tracking_thread.start()
        
        # Start rendering thread
        self.render_thread = threading.Thread(target=self._render_loop)
        self.render_thread.start()
        
        logger.info("AR/VR Core started")
    
    def stop(self):
        """Stop the AR/VR system"""
        self.is_running = False
        
        if self.tracking_thread:
            self.tracking_thread.join()
        
        if self.render_thread:
            self.render_thread.join()
        
        # Disconnect all devices
        for device in self.devices.values():
            if device.is_connected:
                device.disconnect()
        
        logger.info("AR/VR Core stopped")
    
    def _tracking_loop(self):
        """Main tracking update loop"""
        while self.is_running:
            start_time = time.time()
            
            # Update tracking for all connected devices
            for device_id, device in self.devices.items():
                if device.is_connected:
                    try:
                        tracking_data = device.get_tracking_data()
                        self._process_tracking_data(device_id, tracking_data)
                    except Exception as e:
                        logger.error(f"Tracking error for device {device_id}: {e}")
                        if self.callbacks['on_tracking_lost']:
                            self.callbacks['on_tracking_lost'](device_id)
            
            # Calculate tracking latency
            self.metrics['tracking_latency'] = (time.time() - start_time) * 1000
            
            # Maintain 1000Hz tracking rate
            time.sleep(max(0, 0.001 - (time.time() - start_time)))
    
    def _process_tracking_data(self, device_id: str, tracking_data: Dict[str, Any]):
        """Process tracking data from device"""
        if 'transform' in tracking_data:
            transform_data = tracking_data['transform']
            device = self.devices[device_id]
            
            # Update device transform
            if 'position' in transform_data:
                device.transform.position = np.array(transform_data['position'])
            if 'rotation' in transform_data:
                device.transform.rotation = np.array(transform_data['rotation'])
            
            # Update coordinate spaces
            if device_id == self.active_device:
                self.coordinate_transforms[CoordinateSpace.DEVICE.value] = device.transform
    
    def _render_loop(self):
        """Main rendering loop"""
        while self.is_running:
            frame_start = time.time()
            
            if self.active_device and self.active_device in self.devices:
                # Execute render pipeline
                frame_data = None
                for stage in self.render_pipeline:
                    frame_data = stage(frame_data)
                
                # Send to device
                if frame_data is not None:
                    device = self.devices[self.active_device]
                    device.render(frame_data)
            
            # Calculate metrics
            frame_time = time.time() - frame_start
            self.metrics['frame_time'] = frame_time * 1000
            self.metrics['fps'] = 1.0 / frame_time if frame_time > 0 else 0
            
            # Frame rate limiting
            target_frame_time = 1.0 / self.fps_target
            sleep_time = target_frame_time - frame_time
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    # Render pipeline stages
    def _stage_tracking_prediction(self, frame_data: Any) -> Any:
        """Predict future tracking position"""
        if self.config['tracking_prediction_ms'] > 0 and self.active_device:
            # Simple linear prediction
            device = self.devices[self.active_device]
            prediction_time = self.config['tracking_prediction_ms'] / 1000.0
            
            # Would implement velocity-based prediction here
            
        return frame_data
    
    def _stage_culling(self, frame_data: Any) -> Any:
        """Frustum culling stage"""
        # Would implement view frustum culling
        return frame_data
    
    def _stage_rendering(self, frame_data: Any) -> np.ndarray:
        """Main rendering stage"""
        if not self.active_device:
            return np.zeros((self.render_resolution[1], self.render_resolution[0], 3), dtype=np.uint8)
        
        device = self.devices[self.active_device]
        
        # Create frame buffer if needed
        if self.frame_buffer is None or self.frame_buffer.shape[:2] != self.render_resolution:
            self.frame_buffer = np.zeros(
                (self.render_resolution[1], self.render_resolution[0], 4),
                dtype=np.uint8
            )
        
        # Clear frame
        self.frame_buffer.fill(0)
        
        # Render based on mode
        if self.render_mode == RenderMode.STEREO_VR:
            self._render_stereo_vr()
        elif self.render_mode == RenderMode.MONO_AR:
            self._render_mono_ar()
        elif self.render_mode == RenderMode.HOLOGRAPHIC_3D:
            self._render_holographic()
        
        return self.frame_buffer
    
    def _render_stereo_vr(self):
        """Render stereo VR view"""
        # Split frame buffer for left/right eye
        width = self.render_resolution[0] // 2
        
        # Render left eye
        self._render_eye_view(self.frame_buffer[:, :width], -0.032)
        
        # Render right eye
        self._render_eye_view(self.frame_buffer[:, width:], 0.032)
    
    def _render_mono_ar(self):
        """Render mono AR view"""
        # Single view rendering for AR
        self._render_eye_view(self.frame_buffer, 0)
    
    def _render_holographic(self):
        """Render holographic 3D view"""
        # Would implement light field or multi-view rendering
        pass
    
    def _render_eye_view(self, buffer: np.ndarray, eye_offset: float):
        """Render view for a single eye"""
        # Would implement actual 3D rendering here
        # For now, just fill with test pattern
        buffer[:, :, 0] = 64  # Red channel
        buffer[:, :, 1] = 128  # Green channel
        buffer[:, :, 2] = 192  # Blue channel
        buffer[:, :, 3] = 255  # Alpha channel
    
    def _stage_post_processing(self, frame_data: np.ndarray) -> np.ndarray:
        """Post-processing effects"""
        if frame_data is None:
            return None
        
        # Would implement effects like:
        # - Lens distortion correction
        # - Chromatic aberration correction
        # - Anti-aliasing
        # - Color correction
        
        return frame_data
    
    def _stage_reprojection(self, frame_data: np.ndarray) -> np.ndarray:
        """Asynchronous reprojection for motion smoothing"""
        if not self.config['enable_reprojection'] or frame_data is None:
            return frame_data
        
        # Would implement ATW (Asynchronous Time Warp) or
        # ASW (Asynchronous Space Warp) here
        
        return frame_data
    
    def _stage_output(self, frame_data: np.ndarray) -> np.ndarray:
        """Final output stage"""
        # Apply any final transformations before sending to device
        return frame_data
    
    # Coordinate transformation utilities
    def transform_point(self, point: np.ndarray, from_space: CoordinateSpace, 
                       to_space: CoordinateSpace) -> np.ndarray:
        """Transform a point between coordinate spaces"""
        if from_space == to_space:
            return point
        
        # Get transformation matrices
        from_transform = self.coordinate_transforms[from_space.value]
        to_transform = self.coordinate_transforms[to_space.value]
        
        # Convert to homogeneous coordinates
        point_4d = np.append(point, 1)
        
        # Transform to world space first
        world_point = from_transform.matrix @ point_4d
        
        # Then transform to target space
        inv_to_matrix = np.linalg.inv(to_transform.matrix)
        result = inv_to_matrix @ world_point
        
        return result[:3]
    
    def project_to_screen(self, world_point: np.ndarray) -> Optional[Tuple[int, int]]:
        """Project world point to screen coordinates"""
        if not self.active_device:
            return None
        
        device = self.devices[self.active_device]
        if not device.render_target:
            return None
        
        # Transform to device space
        device_point = self.transform_point(
            world_point, 
            CoordinateSpace.WORLD,
            CoordinateSpace.DEVICE
        )
        
        # Apply projection
        rt = device.render_target
        
        # Simple perspective projection
        if device_point[2] <= 0:  # Behind camera
            return None
        
        fov_rad = np.radians(rt.fov)
        aspect = rt.width / rt.height
        
        x = device_point[0] / (device_point[2] * np.tan(fov_rad/2))
        y = device_point[1] / (device_point[2] * np.tan(fov_rad/2) / aspect)
        
        # Convert to screen coordinates
        screen_x = int((x + 1) * rt.width / 2)
        screen_y = int((1 - y) * rt.height / 2)
        
        # Check bounds
        if 0 <= screen_x < rt.width and 0 <= screen_y < rt.height:
            return (screen_x, screen_y)
        
        return None
    
    # Configuration methods
    def set_render_resolution(self, width: int, height: int):
        """Set rendering resolution"""
        self.render_resolution = (width, height)
        self.frame_buffer = None  # Force recreation
    
    def set_config(self, config: Dict[str, Any]):
        """Update system configuration"""
        self.config.update(config)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get performance metrics"""
        self.metrics['total_latency'] = (
            self.metrics['tracking_latency'] + 
            self.metrics['render_latency']
        )
        return self.metrics.copy()
    
    def set_callback(self, event: str, callback: callable):
        """Set event callback"""
        if event in self.callbacks:
            self.callbacks[event] = callback
    
    # Utility methods
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture current rendered frame"""
        if self.frame_buffer is not None:
            return self.frame_buffer.copy()
        return None
    
    def reset_tracking(self):
        """Reset tracking origin"""
        for transform in self.coordinate_transforms.values():
            transform.position = np.zeros(3)
            transform.rotation = np.array([0, 0, 0, 1])
            transform.scale = np.ones(3)
        
        logger.info("Tracking reset to origin")