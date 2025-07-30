"""
Tests for Gesture Recognition system.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch
from collections import deque
from backend.ar_vr_integration.gesture_recognition import (
    GestureRecognizer, GestureType, HandType, HandPose, HandJoint,
    Gesture
)


class TestGestureRecognizer:
    """Test cases for GestureRecognizer class"""
    
    @pytest.fixture
    def recognizer(self):
        """Create gesture recognizer instance for testing"""
        return GestureRecognizer()
    
    def test_initialization(self, recognizer):
        """Test gesture recognizer initialization"""
        assert recognizer.is_running is False
        assert len(recognizer.hand_buffer) == 2  # Left and right
        assert len(recognizer.active_gestures) == 0
        assert len(recognizer.gesture_history) == 0
    
    def test_start_stop(self, recognizer):
        """Test starting and stopping recognizer"""
        recognizer.start()
        assert recognizer.is_running is True
        assert recognizer.process_thread is not None
        assert recognizer.process_thread.is_alive()
        
        recognizer.stop()
        assert recognizer.is_running is False
    
    def test_hand_pose_update(self, recognizer):
        """Test updating hand pose data"""
        joints = {
            'wrist': HandJoint('wrist', np.array([0, 0, 0])),
            'thumb_tip': HandJoint('thumb_tip', np.array([0.05, 0, 0])),
            'index_tip': HandJoint('index_tip', np.array([0.1, 0, 0]))
        }
        
        hand_pose = HandPose(
            hand_type=HandType.RIGHT,
            joints=joints,
            timestamp=time.time(),
            palm_position=np.array([0, 0, 0]),
            palm_normal=np.array([0, 0, 1])
        )
        
        recognizer.update_hand_pose(hand_pose)
        assert not recognizer.input_queue.empty()
    
    def test_pinch_detection(self, recognizer):
        """Test pinch gesture detection"""
        # Create hand poses for pinch
        joints = {
            'wrist': HandJoint('wrist', np.array([0, 0, 0])),
            'thumb_tip': HandJoint('thumb_tip', np.array([0.02, 0, 0])),
            'index_tip': HandJoint('index_tip', np.array([0.025, 0, 0]))  # Close together
        }
        
        hand_pose = HandPose(
            hand_type=HandType.RIGHT,
            joints=joints,
            timestamp=time.time(),
            palm_position=np.array([0, 0, 0]),
            palm_normal=np.array([0, 0, 1])
        )
        
        # Add to buffer
        recognizer.hand_buffer[HandType.RIGHT].append(hand_pose)
        
        # Test pinch detection
        is_pinch = recognizer._detect_pinch(HandType.RIGHT, recognizer.hand_buffer[HandType.RIGHT])
        assert is_pinch is True
    
    def test_grab_detection(self, recognizer):
        """Test grab gesture detection"""
        # Create hand pose for closed fist
        joints = {}
        palm_position = np.array([0, 0, 0])
        
        # Add fingertip joints close to palm
        for finger in ['index', 'middle', 'ring', 'pinky']:
            tip_joint = f'{finger}_tip'
            joints[tip_joint] = HandJoint(tip_joint, palm_position + np.array([0.03, 0, 0]))
        
        hand_pose = HandPose(
            hand_type=HandType.LEFT,
            joints=joints,
            timestamp=time.time(),
            palm_position=palm_position,
            palm_normal=np.array([0, 0, 1]),
            fingers_extended=[False, False, False, False, False]
        )
        
        # Add to buffer
        recognizer.hand_buffer[HandType.LEFT].append(hand_pose)
        
        # Test grab detection
        is_grab = recognizer._detect_grab(HandType.LEFT, recognizer.hand_buffer[HandType.LEFT])
        assert is_grab is True
    
    def test_point_detection(self, recognizer):
        """Test pointing gesture detection"""
        # Create hand pose for pointing (index extended)
        hand_pose = HandPose(
            hand_type=HandType.RIGHT,
            joints={},
            timestamp=time.time(),
            palm_position=np.array([0, 0, 0]),
            palm_normal=np.array([0, 0, 1]),
            fingers_extended=[False, True, False, False, False]  # Only index extended
        )
        
        # Add to buffer
        recognizer.hand_buffer[HandType.RIGHT].append(hand_pose)
        
        # Test point detection
        is_point = recognizer._detect_point(HandType.RIGHT, recognizer.hand_buffer[HandType.RIGHT])
        assert is_point is True
    
    def test_swipe_detection(self, recognizer):
        """Test swipe gesture detection"""
        # Create sequence of hand poses moving horizontally
        positions = np.linspace([0, 0, 0], [0.3, 0, 0], 15)  # 30cm movement
        
        for i, pos in enumerate(positions):
            hand_pose = HandPose(
                hand_type=HandType.RIGHT,
                joints={},
                timestamp=time.time() + i * 0.016,  # 60Hz
                palm_position=pos,
                palm_normal=np.array([0, 0, 1])
            )
            recognizer.hand_buffer[HandType.RIGHT].append(hand_pose)
        
        # Test swipe detection
        swipe_type = recognizer._detect_swipe(HandType.RIGHT, recognizer.hand_buffer[HandType.RIGHT])
        assert swipe_type == GestureType.SWIPE_RIGHT
    
    def test_callback_registration(self, recognizer):
        """Test gesture callback registration"""
        mock_callback = Mock()
        
        recognizer.register_callback(GestureType.PINCH, mock_callback)
        assert mock_callback in recognizer.gesture_callbacks[GestureType.PINCH]
        
        recognizer.unregister_callback(GestureType.PINCH, mock_callback)
        assert mock_callback not in recognizer.gesture_callbacks[GestureType.PINCH]
    
    def test_active_gestures(self, recognizer):
        """Test getting active gestures"""
        # Create a mock gesture
        gesture = Gesture(
            gesture_type=GestureType.PINCH,
            hand_type=HandType.RIGHT,
            start_time=time.time()
        )
        
        recognizer.active_gestures[HandType.RIGHT] = gesture
        
        active = recognizer.get_active_gestures()
        assert HandType.RIGHT in active
        assert active[HandType.RIGHT].gesture_type == GestureType.PINCH
    
    def test_gesture_history(self, recognizer):
        """Test gesture history tracking"""
        # Add some gestures to history
        for i in range(5):
            gesture = Gesture(
                gesture_type=GestureType.TAP,
                hand_type=HandType.LEFT,
                start_time=time.time() - i
            )
            recognizer.gesture_history.append(gesture)
        
        history = recognizer.get_gesture_history(count=3)
        assert len(history) == 3
    
    def test_configuration(self, recognizer):
        """Test configuration updates"""
        config = {
            'pinch_threshold': 0.05,
            'swipe_min_distance': 0.2,
            'confidence_threshold': 0.8
        }
        
        recognizer.set_config(config)
        assert recognizer.config['pinch_threshold'] == 0.05
        assert recognizer.config['swipe_min_distance'] == 0.2
        assert recognizer.config['confidence_threshold'] == 0.8
    
    def test_reset(self, recognizer):
        """Test resetting recognizer state"""
        # Add some data
        recognizer.hand_buffer[HandType.RIGHT].append(Mock())
        recognizer.active_gestures[HandType.RIGHT] = Mock()
        recognizer.gesture_history.append(Mock())
        
        recognizer.reset()
        
        assert len(recognizer.hand_buffer[HandType.RIGHT]) == 0
        assert len(recognizer.hand_buffer[HandType.LEFT]) == 0
        assert len(recognizer.active_gestures) == 0
        assert len(recognizer.gesture_history) == 0


class TestGestureType:
    """Test GestureType enumeration"""
    
    def test_gesture_types(self):
        """Test gesture type values"""
        assert GestureType.NONE.value == "none"
        assert GestureType.POINT.value == "point"
        assert GestureType.PINCH.value == "pinch"
        assert GestureType.GRAB.value == "grab"
        assert GestureType.SWIPE_LEFT.value == "swipe_left"
        assert GestureType.SWIPE_RIGHT.value == "swipe_right"
        assert GestureType.TAP.value == "tap"
        assert GestureType.HOLD.value == "hold"


class TestHandType:
    """Test HandType enumeration"""
    
    def test_hand_types(self):
        """Test hand type values"""
        assert HandType.LEFT.value == "left"
        assert HandType.RIGHT.value == "right"
        assert HandType.BOTH.value == "both"


class TestGesture:
    """Test Gesture dataclass"""
    
    def test_gesture_creation(self):
        """Test creating gesture"""
        gesture = Gesture(
            gesture_type=GestureType.PINCH,
            hand_type=HandType.RIGHT,
            start_time=time.time(),
            confidence=0.9
        )
        
        assert gesture.gesture_type == GestureType.PINCH
        assert gesture.hand_type == HandType.RIGHT
        assert gesture.end_time is None
        assert gesture.is_active is True
        assert gesture.duration > 0
    
    def test_gesture_completion(self):
        """Test completing gesture"""
        start_time = time.time()
        gesture = Gesture(
            gesture_type=GestureType.GRAB,
            hand_type=HandType.LEFT,
            start_time=start_time
        )
        
        time.sleep(0.1)
        gesture.end_time = time.time()
        
        assert gesture.is_active is False
        assert gesture.duration > 0.1