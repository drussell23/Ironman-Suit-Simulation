"""
Eye Tracking Integration for Iron Man suit AR/VR.

This module provides eye tracking capabilities for gaze-based interactions,
foveated rendering, attention monitoring, and hands-free control.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
from collections import deque
import math


class CalibrationState(Enum):
    """Eye tracking calibration states"""
    NOT_CALIBRATED = "not_calibrated"
    CALIBRATING = "calibrating"
    CALIBRATED = "calibrated"
    NEEDS_RECALIBRATION = "needs_recalibration"


class GazeEventType(Enum):
    """Types of gaze events"""
    FIXATION_START = "fixation_start"
    FIXATION_END = "fixation_end"
    SACCADE = "saccade"
    BLINK = "blink"
    SMOOTH_PURSUIT = "smooth_pursuit"
    DWELL = "dwell"  # Extended fixation


@dataclass
class EyeData:
    """Raw eye tracking data"""
    timestamp: float
    left_pupil_position: np.ndarray  # 2D position in camera space
    right_pupil_position: np.ndarray
    left_pupil_diameter: float  # mm
    right_pupil_diameter: float
    left_eye_openness: float  # 0-1
    right_eye_openness: float
    confidence: float  # 0-1


@dataclass
class GazeData:
    """Processed gaze data"""
    timestamp: float
    gaze_origin: np.ndarray  # 3D origin point
    gaze_direction: np.ndarray  # 3D normalized direction
    gaze_point_2d: Tuple[float, float]  # Screen coordinates
    gaze_point_3d: Optional[np.ndarray]  # 3D world intersection
    convergence_distance: float  # Distance where eyes converge
    confidence: float


@dataclass
class Fixation:
    """Fixation event data"""
    start_time: float
    end_time: Optional[float]
    position: np.ndarray  # Average position during fixation
    duration: float = 0.0
    dispersion: float = 0.0  # Spatial variance
    target_id: Optional[str] = None  # ID of object being looked at


@dataclass
class CalibrationPoint:
    """Calibration target point"""
    id: int
    screen_position: Tuple[float, float]  # Normalized 0-1
    samples: List[EyeData] = field(default_factory=list)
    completed: bool = False


class EyeTracker:
    """
    Main eye tracking system for AR/VR interaction.
    
    Provides gaze tracking, fixation detection, calibration,
    and gaze-based control integration.
    """
    
    def __init__(self):
        # Tracking state
        self.is_tracking = False
        self.calibration_state = CalibrationState.NOT_CALIBRATED
        
        # Data buffers
        self.eye_data_buffer = deque(maxlen=120)  # 2 seconds at 60Hz
        self.gaze_data_buffer = deque(maxlen=120)
        
        # Calibration
        self.calibration_points: List[CalibrationPoint] = []
        self.calibration_matrix: Optional[np.ndarray] = None
        self.calibration_quality: float = 0.0
        
        # Event detection
        self.current_fixation: Optional[Fixation] = None
        self.fixation_history = deque(maxlen=100)
        self.last_saccade_time: float = 0.0
        
        # Configuration
        self.config = {
            'sampling_rate': 60,  # Hz
            'fixation_threshold': 1.5,  # degrees
            'fixation_min_duration': 0.1,  # seconds
            'saccade_velocity_threshold': 30,  # degrees/second
            'blink_duration_threshold': 0.3,  # seconds
            'calibration_points': 9,  # 3x3 grid
            'smoothing_window': 5,  # frames
            'pupil_size_filter': True,
            'confidence_threshold': 0.7
        }
        
        # Foveated rendering
        self.foveation_enabled = True
        self.foveation_regions = {
            'foveal': 5,  # degrees - highest quality
            'parafoveal': 15,  # degrees - medium quality  
            'peripheral': 180  # degrees - lowest quality
        }
        
        # Processing thread
        self.processing_thread = None
        self.is_running = False
        
        # Callbacks
        self.callbacks = {
            'on_fixation': None,
            'on_saccade': None,
            'on_blink': None,
            'on_dwell': None,
            'on_calibration_complete': None
        }
        
        # Performance metrics
        self.metrics = {
            'tracking_fps': 0,
            'fixation_rate': 0,
            'saccade_rate': 0,
            'blink_rate': 0,
            'calibration_error': 0,
            'data_loss_rate': 0
        }
    
    def start(self):
        """Start eye tracking"""
        if self.is_tracking:
            return
        
        self.is_tracking = True
        self.is_running = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.start()
    
    def stop(self):
        """Stop eye tracking"""
        self.is_tracking = False
        self.is_running = False
        
        if self.processing_thread:
            self.processing_thread.join()
    
    def start_calibration(self):
        """Start calibration procedure"""
        if self.calibration_state == CalibrationState.CALIBRATING:
            return
        
        self.calibration_state = CalibrationState.CALIBRATING
        self.calibration_points.clear()
        
        # Generate calibration points
        self._generate_calibration_points()
    
    def _generate_calibration_points(self):
        """Generate calibration target points"""
        # 3x3 grid
        grid_size = int(math.sqrt(self.config['calibration_points']))
        
        for i in range(grid_size):
            for j in range(grid_size):
                x = (j + 0.5) / grid_size
                y = (i + 0.5) / grid_size
                
                point = CalibrationPoint(
                    id=i * grid_size + j,
                    screen_position=(x, y)
                )
                self.calibration_points.append(point)
    
    def collect_calibration_sample(self, point_id: int, eye_data: EyeData):
        """Collect sample for calibration point"""
        if (self.calibration_state != CalibrationState.CALIBRATING or
            point_id >= len(self.calibration_points)):
            return
        
        point = self.calibration_points[point_id]
        point.samples.append(eye_data)
        
        # Check if enough samples collected
        if len(point.samples) >= 30:  # 0.5 seconds at 60Hz
            point.completed = True
            
            # Check if all points completed
            if all(p.completed for p in self.calibration_points):
                self._complete_calibration()
    
    def _complete_calibration(self):
        """Complete calibration and compute mapping"""
        # Compute calibration matrix
        self._compute_calibration_matrix()
        
        # Evaluate calibration quality
        self.calibration_quality = self._evaluate_calibration_quality()
        
        if self.calibration_quality > 0.8:
            self.calibration_state = CalibrationState.CALIBRATED
        else:
            self.calibration_state = CalibrationState.NEEDS_RECALIBRATION
        
        # Fire callback
        if self.callbacks['on_calibration_complete']:
            self.callbacks['on_calibration_complete'](self.calibration_quality)
    
    def _compute_calibration_matrix(self):
        """Compute calibration transformation matrix"""
        # Collect all calibration data
        screen_points = []
        eye_points = []
        
        for cal_point in self.calibration_points:
            if not cal_point.completed:
                continue
            
            # Average eye position for this calibration point
            avg_left = np.mean([s.left_pupil_position for s in cal_point.samples], axis=0)
            avg_right = np.mean([s.right_pupil_position for s in cal_point.samples], axis=0)
            avg_eye = (avg_left + avg_right) / 2
            
            screen_points.append(cal_point.screen_position)
            eye_points.append(avg_eye)
        
        if len(screen_points) < 4:
            return
        
        # Compute homography matrix
        screen_points = np.array(screen_points)
        eye_points = np.array(eye_points)
        
        # Simple linear mapping for now
        # Would use proper homography estimation
        self.calibration_matrix = np.eye(3)
    
    def _evaluate_calibration_quality(self) -> float:
        """Evaluate calibration quality"""
        if self.calibration_matrix is None:
            return 0.0
        
        # Calculate average prediction error
        errors = []
        
        for cal_point in self.calibration_points:
            if not cal_point.completed:
                continue
            
            for sample in cal_point.samples[-10:]:  # Last 10 samples
                # Predict gaze point
                predicted = self._transform_eye_to_screen(
                    (sample.left_pupil_position + sample.right_pupil_position) / 2
                )
                
                # Calculate error
                actual = cal_point.screen_position
                error = math.sqrt((predicted[0] - actual[0])**2 + 
                                (predicted[1] - actual[1])**2)
                errors.append(error)
        
        if not errors:
            return 0.0
        
        # Convert error to quality score
        avg_error = np.mean(errors)
        quality = max(0, 1 - avg_error * 5)  # 0.2 error = 0 quality
        
        return quality
    
    def _processing_loop(self):
        """Main processing loop"""
        last_time = time.time()
        
        while self.is_running:
            current_time = time.time()
            
            # Process latest eye data
            if self.eye_data_buffer:
                eye_data = self.eye_data_buffer[-1]
                
                # Convert to gaze data
                gaze_data = self._process_eye_data(eye_data)
                
                if gaze_data:
                    self.gaze_data_buffer.append(gaze_data)
                    
                    # Detect events
                    self._detect_gaze_events(gaze_data)
            
            # Update metrics
            self.metrics['tracking_fps'] = 1.0 / (current_time - last_time)
            last_time = current_time
            
            # Maintain sampling rate
            sleep_time = 1.0 / self.config['sampling_rate'] - (time.time() - current_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _process_eye_data(self, eye_data: EyeData) -> Optional[GazeData]:
        """Process raw eye data into gaze data"""
        # Check data quality
        if eye_data.confidence < self.config['confidence_threshold']:
            return None
        
        # Check for blink
        if (eye_data.left_eye_openness < 0.2 and 
            eye_data.right_eye_openness < 0.2):
            self._handle_blink()
            return None
        
        # Calculate gaze from pupil positions
        if self.calibration_state == CalibrationState.CALIBRATED:
            # Transform to screen coordinates
            left_screen = self._transform_eye_to_screen(eye_data.left_pupil_position)
            right_screen = self._transform_eye_to_screen(eye_data.right_pupil_position)
            
            # Average for combined gaze
            gaze_point_2d = (
                (left_screen[0] + right_screen[0]) / 2,
                (left_screen[1] + right_screen[1]) / 2
            )
        else:
            # Raw gaze without calibration
            gaze_point_2d = (0.5, 0.5)  # Center
        
        # Calculate 3D gaze ray
        gaze_origin, gaze_direction = self._calculate_3d_gaze(eye_data)
        
        # Calculate convergence distance
        convergence_distance = self._calculate_convergence_distance(eye_data)
        
        # Apply smoothing
        if self.config['smoothing_window'] > 1 and len(self.gaze_data_buffer) > 0:
            gaze_point_2d = self._smooth_gaze_point(gaze_point_2d)
        
        return GazeData(
            timestamp=eye_data.timestamp,
            gaze_origin=gaze_origin,
            gaze_direction=gaze_direction,
            gaze_point_2d=gaze_point_2d,
            gaze_point_3d=None,  # Set by raycast
            convergence_distance=convergence_distance,
            confidence=eye_data.confidence
        )
    
    def _transform_eye_to_screen(self, eye_position: np.ndarray) -> Tuple[float, float]:
        """Transform eye position to screen coordinates"""
        if self.calibration_matrix is None:
            return (0.5, 0.5)
        
        # Apply calibration transformation
        # Simplified linear mapping
        x = eye_position[0] * 0.5 + 0.5
        y = eye_position[1] * 0.5 + 0.5
        
        return (np.clip(x, 0, 1), np.clip(y, 0, 1))
    
    def _calculate_3d_gaze(self, eye_data: EyeData) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate 3D gaze ray from eye data"""
        # Simplified model - would use proper eye model
        
        # Average eye position (interpupillary distance ~63mm)
        ipd = 0.063  # meters
        left_eye_pos = np.array([-ipd/2, 0, 0])
        right_eye_pos = np.array([ipd/2, 0, 0])
        
        # Gaze origin is between eyes
        gaze_origin = (left_eye_pos + right_eye_pos) / 2
        
        # Calculate gaze direction from pupil positions
        # This is simplified - real implementation would use eye model
        left_dir = np.array([
            eye_data.left_pupil_position[0],
            eye_data.left_pupil_position[1],
            1.0
        ])
        right_dir = np.array([
            eye_data.right_pupil_position[0],
            eye_data.right_pupil_position[1],
            1.0
        ])
        
        # Average and normalize
        gaze_direction = (left_dir + right_dir) / 2
        gaze_direction = gaze_direction / np.linalg.norm(gaze_direction)
        
        return gaze_origin, gaze_direction
    
    def _calculate_convergence_distance(self, eye_data: EyeData) -> float:
        """Calculate distance where both eyes converge"""
        # Simplified vergence calculation
        # Would use proper binocular vision model
        
        # Estimate from pupil positions
        left_x = eye_data.left_pupil_position[0]
        right_x = eye_data.right_pupil_position[0]
        
        # Convergence angle
        convergence_angle = abs(left_x - right_x) * 0.1  # Scaling factor
        
        if convergence_angle > 0:
            # Distance = IPD / (2 * tan(angle/2))
            ipd = 0.063
            distance = ipd / (2 * math.tan(convergence_angle / 2))
            return np.clip(distance, 0.1, 10.0)  # Clamp to reasonable range
        
        return 2.0  # Default 2 meters
    
    def _smooth_gaze_point(self, current_point: Tuple[float, float]) -> Tuple[float, float]:
        """Apply smoothing to gaze point"""
        window_size = min(self.config['smoothing_window'], len(self.gaze_data_buffer))
        
        if window_size < 2:
            return current_point
        
        # Get recent points
        recent_points = [g.gaze_point_2d for g in list(self.gaze_data_buffer)[-window_size:]]
        recent_points.append(current_point)
        
        # Weighted average (more weight on recent)
        weights = np.linspace(0.5, 1.0, len(recent_points))
        weights = weights / weights.sum()
        
        x = sum(p[0] * w for p, w in zip(recent_points, weights))
        y = sum(p[1] * w for p, w in zip(recent_points, weights))
        
        return (x, y)
    
    def _detect_gaze_events(self, gaze_data: GazeData):
        """Detect gaze events (fixations, saccades, etc.)"""
        if len(self.gaze_data_buffer) < 2:
            return
        
        prev_gaze = self.gaze_data_buffer[-2]
        
        # Calculate gaze velocity
        dt = gaze_data.timestamp - prev_gaze.timestamp
        if dt <= 0:
            return
        
        # Angular velocity in visual degrees
        dx = gaze_data.gaze_point_2d[0] - prev_gaze.gaze_point_2d[0]
        dy = gaze_data.gaze_point_2d[1] - prev_gaze.gaze_point_2d[1]
        
        # Convert to visual angles (assuming ~30 degrees FOV for screen)
        fov = 30
        angular_distance = math.sqrt(dx**2 + dy**2) * fov
        angular_velocity = angular_distance / dt
        
        # Detect saccade
        if angular_velocity > self.config['saccade_velocity_threshold']:
            self._handle_saccade(gaze_data, angular_velocity)
        # Detect fixation
        elif angular_distance < self.config['fixation_threshold']:
            self._handle_fixation(gaze_data)
        # Smooth pursuit
        elif 5 < angular_velocity < self.config['saccade_velocity_threshold']:
            self._handle_smooth_pursuit(gaze_data, angular_velocity)
    
    def _handle_fixation(self, gaze_data: GazeData):
        """Handle fixation detection"""
        if self.current_fixation is None:
            # Start new fixation
            self.current_fixation = Fixation(
                start_time=gaze_data.timestamp,
                end_time=None,
                position=np.array(gaze_data.gaze_point_2d)
            )
            
            if self.callbacks['on_fixation']:
                self.callbacks['on_fixation']('start', self.current_fixation)
        else:
            # Update current fixation
            self.current_fixation.duration = gaze_data.timestamp - self.current_fixation.start_time
            
            # Update average position
            alpha = 0.1  # Smoothing factor
            self.current_fixation.position = (
                (1 - alpha) * self.current_fixation.position +
                alpha * np.array(gaze_data.gaze_point_2d)
            )
            
            # Check for dwell (extended fixation)
            if self.current_fixation.duration > 1.0:  # 1 second
                if self.callbacks['on_dwell']:
                    self.callbacks['on_dwell'](self.current_fixation)
    
    def _handle_saccade(self, gaze_data: GazeData, velocity: float):
        """Handle saccade detection"""
        # End current fixation
        if self.current_fixation:
            self.current_fixation.end_time = gaze_data.timestamp
            self.fixation_history.append(self.current_fixation)
            
            if self.callbacks['on_fixation']:
                self.callbacks['on_fixation']('end', self.current_fixation)
            
            self.current_fixation = None
        
        # Fire saccade callback
        if self.callbacks['on_saccade']:
            self.callbacks['on_saccade'](gaze_data, velocity)
        
        self.last_saccade_time = gaze_data.timestamp
        
        # Update metrics
        self.metrics['saccade_rate'] = len([g for g in self.gaze_data_buffer 
                                           if g.timestamp > time.time() - 60]) / 60
    
    def _handle_smooth_pursuit(self, gaze_data: GazeData, velocity: float):
        """Handle smooth pursuit eye movement"""
        # Smooth pursuit typically occurs when tracking moving objects
        # Would implement more sophisticated detection
        pass
    
    def _handle_blink(self):
        """Handle blink detection"""
        current_time = time.time()
        
        if self.callbacks['on_blink']:
            self.callbacks['on_blink'](current_time)
        
        # Update blink rate
        recent_blinks = [e for e in self.eye_data_buffer 
                        if e.timestamp > current_time - 60 and
                        e.left_eye_openness < 0.2 and e.right_eye_openness < 0.2]
        self.metrics['blink_rate'] = len(recent_blinks)
    
    # Public API methods
    def update_eye_data(self, eye_data: EyeData):
        """Update with new eye tracking data"""
        self.eye_data_buffer.append(eye_data)
    
    def get_current_gaze(self) -> Optional[GazeData]:
        """Get current gaze data"""
        if self.gaze_data_buffer:
            return self.gaze_data_buffer[-1]
        return None
    
    def get_fixation_map(self, duration: float = 10.0) -> List[Dict[str, Any]]:
        """Get recent fixation map"""
        current_time = time.time()
        recent_fixations = [f for f in self.fixation_history 
                          if f.start_time > current_time - duration]
        
        fixation_map = []
        for fixation in recent_fixations:
            fixation_map.append({
                'position': fixation.position.tolist(),
                'duration': fixation.duration,
                'start_time': fixation.start_time,
                'target_id': fixation.target_id
            })
        
        return fixation_map
    
    def get_attention_heatmap(self, resolution: Tuple[int, int] = (64, 64)) -> np.ndarray:
        """Generate attention heatmap from gaze data"""
        heatmap = np.zeros(resolution)
        
        # Accumulate gaze points
        for gaze in self.gaze_data_buffer:
            x = int(gaze.gaze_point_2d[0] * resolution[0])
            y = int(gaze.gaze_point_2d[1] * resolution[1])
            
            if 0 <= x < resolution[0] and 0 <= y < resolution[1]:
                heatmap[y, x] += 1
        
        # Apply Gaussian blur for smooth heatmap
        # Would use proper convolution
        
        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap
    
    def get_foveation_parameters(self) -> Dict[str, Any]:
        """Get parameters for foveated rendering"""
        if not self.foveation_enabled or not self.gaze_data_buffer:
            return {
                'enabled': False,
                'center': (0.5, 0.5),
                'regions': self.foveation_regions
            }
        
        current_gaze = self.gaze_data_buffer[-1]
        
        return {
            'enabled': True,
            'center': current_gaze.gaze_point_2d,
            'regions': self.foveation_regions,
            'confidence': current_gaze.confidence
        }
    
    def perform_gaze_selection(self, targets: List[Dict[str, Any]], 
                             dwell_time: float = 0.8) -> Optional[str]:
        """Perform selection based on gaze dwell time"""
        if not self.current_fixation or self.current_fixation.duration < dwell_time:
            return None
        
        gaze_pos = self.current_fixation.position
        
        # Check which target is being looked at
        for target in targets:
            target_pos = np.array(target['position'])
            target_size = target.get('size', 0.1)
            
            distance = np.linalg.norm(gaze_pos - target_pos)
            
            if distance < target_size:
                return target['id']
        
        return None
    
    def calibrate_ipd(self, ipd_mm: float):
        """Calibrate interpupillary distance"""
        # Store IPD for 3D gaze calculations
        self.ipd = ipd_mm / 1000.0  # Convert to meters
    
    def set_config(self, config: Dict[str, Any]):
        """Update configuration"""
        self.config.update(config)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get eye tracking metrics"""
        # Calculate additional metrics
        if self.gaze_data_buffer:
            total_samples = len(self.gaze_data_buffer)
            valid_samples = sum(1 for g in self.gaze_data_buffer if g.confidence > 0.7)
            self.metrics['data_loss_rate'] = 1 - (valid_samples / total_samples)
        
        self.metrics['calibration_error'] = 1 - self.calibration_quality
        
        # Calculate fixation rate
        recent_fixations = [f for f in self.fixation_history 
                          if f.start_time > time.time() - 60]
        self.metrics['fixation_rate'] = len(recent_fixations) / 60
        
        return self.metrics.copy()
    
    def set_callback(self, event: str, callback: Callable):
        """Set event callback"""
        if event in self.callbacks:
            self.callbacks[event] = callback