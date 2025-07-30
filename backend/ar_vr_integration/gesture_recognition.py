"""
Gesture Recognition system for Iron Man suit AR/VR interface.

This module provides real-time hand gesture recognition for controlling
the suit's virtual interfaces, similar to Tony Stark's gesture-based
holographic manipulation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import deque
import threading
import queue


class GestureType(Enum):
    """Recognized gesture types"""
    NONE = "none"
    POINT = "point"
    PINCH = "pinch"
    GRAB = "grab"
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    SWIPE_UP = "swipe_up"
    SWIPE_DOWN = "swipe_down"
    ROTATE_CW = "rotate_clockwise"
    ROTATE_CCW = "rotate_counter_clockwise"
    SPREAD = "spread"
    PUSH = "push"
    PULL = "pull"
    TAP = "tap"
    DOUBLE_TAP = "double_tap"
    HOLD = "hold"
    RELEASE = "release"


class HandType(Enum):
    """Hand identification"""
    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"


@dataclass
class HandJoint:
    """Individual hand joint data"""
    name: str
    position: np.ndarray  # 3D position
    confidence: float = 1.0
    
    
@dataclass
class HandPose:
    """Complete hand pose data"""
    hand_type: HandType
    joints: Dict[str, HandJoint]
    timestamp: float
    palm_position: np.ndarray
    palm_normal: np.ndarray
    fingers_extended: List[bool] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.fingers_extended:
            self.fingers_extended = [False] * 5  # thumb, index, middle, ring, pinky


@dataclass
class Gesture:
    """Recognized gesture data"""
    gesture_type: GestureType
    hand_type: HandType
    start_time: float
    end_time: Optional[float] = None
    confidence: float = 1.0
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    @property
    def is_active(self) -> bool:
        return self.end_time is None


class GestureRecognizer:
    """
    Main gesture recognition system.
    
    Processes hand tracking data to recognize gestures for
    controlling virtual interfaces.
    """
    
    # Standard hand joint names
    JOINT_NAMES = [
        "wrist",
        "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
        "index_mcp", "index_pip", "index_dip", "index_tip",
        "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
        "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
        "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
    ]
    
    def __init__(self):
        # Hand tracking buffers
        self.hand_buffer: Dict[HandType, deque] = {
            HandType.LEFT: deque(maxlen=60),  # 1 second at 60fps
            HandType.RIGHT: deque(maxlen=60)
        }
        
        # Gesture detection state
        self.active_gestures: Dict[HandType, Gesture] = {}
        self.gesture_history: deque = deque(maxlen=100)
        
        # Recognition parameters
        self.config = {
            'pinch_threshold': 0.03,  # meters
            'grab_threshold': 0.05,  # meters
            'swipe_min_distance': 0.15,  # meters
            'swipe_min_velocity': 0.5,  # m/s
            'rotation_min_angle': 30,  # degrees
            'hold_duration': 0.5,  # seconds
            'double_tap_interval': 0.3,  # seconds
            'confidence_threshold': 0.7
        }
        
        # Processing state
        self.is_running = False
        self.process_thread = None
        self.input_queue = queue.Queue()
        
        # Callbacks
        self.gesture_callbacks: Dict[GestureType, List[Callable]] = {
            gesture_type: [] for gesture_type in GestureType
        }
        
        # Gesture templates for matching
        self._initialize_gesture_templates()
    
    def _initialize_gesture_templates(self):
        """Initialize gesture recognition templates"""
        self.gesture_templates = {
            GestureType.PINCH: self._detect_pinch,
            GestureType.GRAB: self._detect_grab,
            GestureType.POINT: self._detect_point,
            GestureType.SWIPE_LEFT: self._detect_swipe,
            GestureType.SWIPE_RIGHT: self._detect_swipe,
            GestureType.SWIPE_UP: self._detect_swipe,
            GestureType.SWIPE_DOWN: self._detect_swipe,
            GestureType.ROTATE_CW: self._detect_rotation,
            GestureType.ROTATE_CCW: self._detect_rotation,
            GestureType.SPREAD: self._detect_spread,
            GestureType.PUSH: self._detect_push_pull,
            GestureType.PULL: self._detect_push_pull,
            GestureType.TAP: self._detect_tap,
            GestureType.HOLD: self._detect_hold
        }
    
    def start(self):
        """Start gesture recognition"""
        if self.is_running:
            return
        
        self.is_running = True
        self.process_thread = threading.Thread(target=self._process_loop)
        self.process_thread.start()
    
    def stop(self):
        """Stop gesture recognition"""
        self.is_running = False
        if self.process_thread:
            self.process_thread.join()
    
    def update_hand_pose(self, hand_pose: HandPose):
        """Update hand tracking data"""
        self.input_queue.put(hand_pose)
    
    def _process_loop(self):
        """Main processing loop"""
        while self.is_running:
            try:
                # Get hand pose with timeout
                hand_pose = self.input_queue.get(timeout=0.016)  # ~60fps
                
                # Add to buffer
                self.hand_buffer[hand_pose.hand_type].append(hand_pose)
                
                # Process gestures
                self._process_hand_gestures(hand_pose.hand_type)
                
            except queue.Empty:
                # No new data, check for timeouts
                self._check_gesture_timeouts()
            except Exception as e:
                print(f"Gesture processing error: {e}")
    
    def _process_hand_gestures(self, hand_type: HandType):
        """Process gestures for a specific hand"""
        buffer = self.hand_buffer[hand_type]
        
        if len(buffer) < 3:  # Need minimum history
            return
        
        current_pose = buffer[-1]
        
        # Check each gesture type
        for gesture_type, detector in self.gesture_templates.items():
            if detector(hand_type, buffer):
                self._trigger_gesture(gesture_type, hand_type, current_pose)
    
    def _trigger_gesture(self, gesture_type: GestureType, hand_type: HandType, 
                        current_pose: HandPose):
        """Trigger a recognized gesture"""
        # Check if gesture is already active
        if hand_type in self.active_gestures:
            active = self.active_gestures[hand_type]
            if active.gesture_type == gesture_type:
                # Update existing gesture
                return
            else:
                # End previous gesture
                self._end_gesture(hand_type)
        
        # Create new gesture
        gesture = Gesture(
            gesture_type=gesture_type,
            hand_type=hand_type,
            start_time=current_pose.timestamp,
            confidence=self._calculate_confidence(gesture_type, hand_type)
        )
        
        # Add gesture-specific parameters
        self._add_gesture_parameters(gesture, current_pose)
        
        self.active_gestures[hand_type] = gesture
        self.gesture_history.append(gesture)
        
        # Trigger callbacks
        self._fire_callbacks(gesture)
    
    def _end_gesture(self, hand_type: HandType):
        """End an active gesture"""
        if hand_type in self.active_gestures:
            gesture = self.active_gestures[hand_type]
            gesture.end_time = time.time()
            
            # Special handling for release gesture
            if gesture.gesture_type in [GestureType.GRAB, GestureType.PINCH, GestureType.HOLD]:
                release_gesture = Gesture(
                    gesture_type=GestureType.RELEASE,
                    hand_type=hand_type,
                    start_time=gesture.end_time
                )
                self._fire_callbacks(release_gesture)
            
            del self.active_gestures[hand_type]
    
    def _check_gesture_timeouts(self):
        """Check for gesture timeouts"""
        current_time = time.time()
        
        for hand_type in list(self.active_gestures.keys()):
            gesture = self.active_gestures[hand_type]
            
            # Check for hold timeout
            if gesture.gesture_type != GestureType.HOLD and gesture.duration > self.config['hold_duration']:
                # Convert to hold
                gesture.gesture_type = GestureType.HOLD
                self._fire_callbacks(gesture)
    
    # Gesture detection methods
    def _detect_pinch(self, hand_type: HandType, buffer: deque) -> bool:
        """Detect pinch gesture (thumb and index finger together)"""
        current_pose = buffer[-1]
        
        if "thumb_tip" not in current_pose.joints or "index_tip" not in current_pose.joints:
            return False
        
        thumb_tip = current_pose.joints["thumb_tip"].position
        index_tip = current_pose.joints["index_tip"].position
        
        distance = np.linalg.norm(thumb_tip - index_tip)
        
        return distance < self.config['pinch_threshold']
    
    def _detect_grab(self, hand_type: HandType, buffer: deque) -> bool:
        """Detect grab gesture (closed fist)"""
        current_pose = buffer[-1]
        
        # Check if all fingers are bent
        extended_count = sum(current_pose.fingers_extended)
        
        if extended_count > 1:  # More than one finger extended
            return False
        
        # Check palm to fingertip distances
        palm_pos = current_pose.palm_position
        
        for finger in ["index", "middle", "ring", "pinky"]:
            tip_joint = f"{finger}_tip"
            if tip_joint in current_pose.joints:
                tip_pos = current_pose.joints[tip_joint].position
                distance = np.linalg.norm(palm_pos - tip_pos)
                
                if distance > self.config['grab_threshold']:
                    return False
        
        return True
    
    def _detect_point(self, hand_type: HandType, buffer: deque) -> bool:
        """Detect pointing gesture (index finger extended)"""
        current_pose = buffer[-1]
        
        # Check finger extension pattern
        if len(current_pose.fingers_extended) >= 5:
            # Only index finger should be extended
            expected = [False, True, False, False, False]
            return current_pose.fingers_extended == expected
        
        return False
    
    def _detect_swipe(self, hand_type: HandType, buffer: deque) -> bool:
        """Detect swipe gestures"""
        if len(buffer) < 10:
            return False
        
        # Get palm positions over time
        positions = [pose.palm_position for pose in list(buffer)[-10:]]
        
        # Calculate movement vector
        start_pos = positions[0]
        end_pos = positions[-1]
        movement = end_pos - start_pos
        distance = np.linalg.norm(movement)
        
        if distance < self.config['swipe_min_distance']:
            return False
        
        # Calculate velocity
        time_span = buffer[-1].timestamp - buffer[-10].timestamp
        if time_span <= 0:
            return False
        
        velocity = distance / time_span
        
        if velocity < self.config['swipe_min_velocity']:
            return False
        
        # Determine swipe direction
        movement_norm = movement / distance
        
        # Check primary axis
        abs_movement = np.abs(movement_norm)
        primary_axis = np.argmax(abs_movement)
        
        if primary_axis == 0:  # X-axis
            if movement_norm[0] > 0.7:
                return GestureType.SWIPE_RIGHT
            elif movement_norm[0] < -0.7:
                return GestureType.SWIPE_LEFT
        elif primary_axis == 1:  # Y-axis
            if movement_norm[1] > 0.7:
                return GestureType.SWIPE_UP
            elif movement_norm[1] < -0.7:
                return GestureType.SWIPE_DOWN
        
        return False
    
    def _detect_rotation(self, hand_type: HandType, buffer: deque) -> bool:
        """Detect rotation gestures"""
        if len(buffer) < 15:
            return False
        
        # Track wrist rotation
        positions = []
        for i in range(0, 15, 3):
            pose = buffer[-(15-i)]
            if "index_mcp" in pose.joints:
                positions.append(pose.joints["index_mcp"].position - pose.palm_position)
        
        if len(positions) < 3:
            return False
        
        # Calculate rotation angle
        v1 = positions[0]
        v2 = positions[-1]
        
        # Project onto palm plane
        normal = buffer[-1].palm_normal
        v1_proj = v1 - np.dot(v1, normal) * normal
        v2_proj = v2 - np.dot(v2, normal) * normal
        
        # Calculate angle
        cos_angle = np.dot(v1_proj, v2_proj) / (np.linalg.norm(v1_proj) * np.linalg.norm(v2_proj))
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        
        if angle < self.config['rotation_min_angle']:
            return False
        
        # Determine direction using cross product
        cross = np.cross(v1_proj, v2_proj)
        direction = np.dot(cross, normal)
        
        if direction > 0:
            return GestureType.ROTATE_CW
        else:
            return GestureType.ROTATE_CCW
    
    def _detect_spread(self, hand_type: HandType, buffer: deque) -> bool:
        """Detect spread fingers gesture"""
        current_pose = buffer[-1]
        
        # All fingers should be extended
        if sum(current_pose.fingers_extended) < 5:
            return False
        
        # Check finger separation
        finger_tips = []
        for finger in ["thumb", "index", "middle", "ring", "pinky"]:
            tip_joint = f"{finger}_tip"
            if tip_joint in current_pose.joints:
                finger_tips.append(current_pose.joints[tip_joint].position)
        
        if len(finger_tips) < 5:
            return False
        
        # Calculate average separation
        total_separation = 0
        count = 0
        
        for i in range(len(finger_tips)):
            for j in range(i + 1, len(finger_tips)):
                separation = np.linalg.norm(finger_tips[i] - finger_tips[j])
                total_separation += separation
                count += 1
        
        avg_separation = total_separation / count if count > 0 else 0
        
        # Threshold for spread detection
        return avg_separation > 0.06  # 6cm average separation
    
    def _detect_push_pull(self, hand_type: HandType, buffer: deque) -> bool:
        """Detect push/pull gestures"""
        if len(buffer) < 10:
            return False
        
        # Check Z-axis movement
        positions = [pose.palm_position for pose in list(buffer)[-10:]]
        
        start_z = positions[0][2]
        end_z = positions[-1][2]
        z_change = end_z - start_z
        
        if abs(z_change) < 0.1:  # 10cm minimum
            return False
        
        # Calculate velocity
        time_span = buffer[-1].timestamp - buffer[-10].timestamp
        if time_span <= 0:
            return False
        
        velocity = abs(z_change) / time_span
        
        if velocity < 0.3:  # 0.3 m/s minimum
            return False
        
        if z_change > 0:
            return GestureType.PUSH
        else:
            return GestureType.PULL
    
    def _detect_tap(self, hand_type: HandType, buffer: deque) -> bool:
        """Detect tap gesture"""
        if len(buffer) < 5:
            return False
        
        # Look for quick forward-backward movement
        positions = [pose.palm_position for pose in list(buffer)[-5:]]
        
        # Check for movement pattern
        z_values = [pos[2] for pos in positions]
        
        # Find peak (most forward position)
        peak_idx = np.argmin(z_values)
        
        if peak_idx == 0 or peak_idx == len(z_values) - 1:
            return False
        
        # Check for tap pattern
        forward_movement = z_values[0] - z_values[peak_idx]
        backward_movement = z_values[peak_idx] - z_values[-1]
        
        if forward_movement > 0.03 and backward_movement < -0.02:
            # Check for double tap
            recent_taps = [g for g in self.gesture_history 
                          if g.gesture_type == GestureType.TAP and 
                          g.hand_type == hand_type and
                          time.time() - g.start_time < self.config['double_tap_interval']]
            
            if recent_taps:
                return GestureType.DOUBLE_TAP
            else:
                return GestureType.TAP
        
        return False
    
    def _detect_hold(self, hand_type: HandType, buffer: deque) -> bool:
        """Detect hold gesture (handled in timeout check)"""
        return False
    
    def _calculate_confidence(self, gesture_type: GestureType, hand_type: HandType) -> float:
        """Calculate gesture recognition confidence"""
        # Would implement ML-based confidence scoring
        return 0.9
    
    def _add_gesture_parameters(self, gesture: Gesture, current_pose: HandPose):
        """Add gesture-specific parameters"""
        if gesture.gesture_type in [GestureType.SWIPE_LEFT, GestureType.SWIPE_RIGHT, 
                                   GestureType.SWIPE_UP, GestureType.SWIPE_DOWN]:
            # Add swipe direction and distance
            buffer = self.hand_buffer[gesture.hand_type]
            if len(buffer) >= 10:
                start_pos = buffer[-10].palm_position
                end_pos = current_pose.palm_position
                gesture.parameters['direction'] = end_pos - start_pos
                gesture.parameters['distance'] = np.linalg.norm(end_pos - start_pos)
        
        elif gesture.gesture_type == GestureType.PINCH:
            # Add pinch position
            if "thumb_tip" in current_pose.joints and "index_tip" in current_pose.joints:
                thumb_pos = current_pose.joints["thumb_tip"].position
                index_pos = current_pose.joints["index_tip"].position
                gesture.parameters['position'] = (thumb_pos + index_pos) / 2
        
        elif gesture.gesture_type == GestureType.POINT:
            # Add pointing direction
            if "index_tip" in current_pose.joints and "index_mcp" in current_pose.joints:
                tip_pos = current_pose.joints["index_tip"].position
                mcp_pos = current_pose.joints["index_mcp"].position
                direction = tip_pos - mcp_pos
                gesture.parameters['direction'] = direction / np.linalg.norm(direction)
    
    def _fire_callbacks(self, gesture: Gesture):
        """Fire callbacks for recognized gesture"""
        for callback in self.gesture_callbacks[gesture.gesture_type]:
            try:
                callback(gesture)
            except Exception as e:
                print(f"Gesture callback error: {e}")
    
    # Public API
    def register_callback(self, gesture_type: GestureType, callback: Callable):
        """Register callback for gesture type"""
        self.gesture_callbacks[gesture_type].append(callback)
    
    def unregister_callback(self, gesture_type: GestureType, callback: Callable):
        """Unregister callback for gesture type"""
        if callback in self.gesture_callbacks[gesture_type]:
            self.gesture_callbacks[gesture_type].remove(callback)
    
    def get_active_gestures(self) -> Dict[HandType, Gesture]:
        """Get currently active gestures"""
        return self.active_gestures.copy()
    
    def get_gesture_history(self, count: int = 10) -> List[Gesture]:
        """Get recent gesture history"""
        return list(self.gesture_history)[-count:]
    
    def set_config(self, config: Dict[str, Any]):
        """Update configuration parameters"""
        self.config.update(config)
    
    def calibrate_hand(self, hand_type: HandType, calibration_data: Dict[str, Any]):
        """Calibrate hand-specific parameters"""
        # Would store hand-specific calibration
        pass
    
    def reset(self):
        """Reset gesture recognition state"""
        self.hand_buffer[HandType.LEFT].clear()
        self.hand_buffer[HandType.RIGHT].clear()
        self.active_gestures.clear()
        self.gesture_history.clear()