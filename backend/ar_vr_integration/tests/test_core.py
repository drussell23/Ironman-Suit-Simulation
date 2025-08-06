"""
Tests for AR/VR Core framework.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, MagicMock
from backend.ar_vr_integration.core import (
    ARVRCore, DeviceType, RenderMode, CoordinateSpace, Transform,
    ARVRDevice, RenderTarget
)


class MockARVRDevice(ARVRDevice):
    """Mock AR/VR device for testing"""
    
    def connect(self) -> bool:
        return True
    
    def disconnect(self):
        pass
    
    def get_tracking_data(self):
        return {
            'transform': {
                'position': [0, 0, 0],
                'rotation': [0, 0, 0, 1]
            }
        }
    
    def render(self, frame_data):
        pass


class TestTransform:
    """Test Transform class"""
    
    def test_initialization(self):
        """Test transform initialization"""
        transform = Transform()
        assert np.array_equal(transform.position, np.zeros(3))
        assert np.array_equal(transform.rotation, np.array([0, 0, 0, 1]))
        assert np.array_equal(transform.scale, np.ones(3))
    
    def test_matrix_property(self):
        """Test transformation matrix generation"""
        transform = Transform()
        transform.position = np.array([1, 2, 3])
        
        matrix = transform.matrix
        assert matrix.shape == (4, 4)
        assert np.array_equal(matrix[:3, 3], [1, 2, 3])
        assert matrix[3, 3] == 1


class TestARVRCore:
    """Test cases for ARVRCore class"""
    
    @pytest.fixture
    def core(self):
        """Create AR/VR core instance for testing"""
        return ARVRCore()
    
    def test_initialization(self, core):
        """Test core initialization"""
        assert len(core.devices) == 0
        assert core.active_device is None
        assert core.render_mode == RenderMode.MONO_AR
        assert core.is_running is False
        assert len(core.render_pipeline) > 0
    
    def test_add_remove_device(self, core):
        """Test adding and removing devices"""
        device = MockARVRDevice("test_device", DeviceType.HMD_VR)
        
        # Add device
        result = core.add_device(device)
        assert result is True
        assert "test_device" in core.devices
        assert core.active_device == "test_device"
        
        # Remove device
        core.remove_device("test_device")
        assert "test_device" not in core.devices
        assert core.active_device is None
    
    def test_set_active_device(self, core):
        """Test setting active device"""
        device1 = MockARVRDevice("device1", DeviceType.HMD_VR)
        device2 = MockARVRDevice("device2", DeviceType.HMD_AR)
        
        core.add_device(device1)
        core.add_device(device2)
        
        # Set device2 as active
        result = core.set_active_device("device2")
        assert result is True
        assert core.active_device == "device2"
        assert core.render_mode == RenderMode.MONO_AR
    
    def test_render_mode_update(self, core):
        """Test render mode updates based on device type"""
        # VR device
        vr_device = MockARVRDevice("vr_device", DeviceType.HMD_VR)
        core.add_device(vr_device)
        assert core.render_mode == RenderMode.STEREO_VR
        
        # AR device
        ar_device = MockARVRDevice("ar_device", DeviceType.HMD_AR)
        core.add_device(ar_device)
        core.set_active_device("ar_device")
        assert core.render_mode == RenderMode.MONO_AR
    
    def test_start_stop(self, core):
        """Test starting and stopping core system"""
        device = MockARVRDevice("test_device", DeviceType.HMD_VR)
        core.add_device(device)
        
        core.start()
        assert core.is_running is True
        assert core.tracking_thread is not None
        assert core.render_thread is not None
        
        time.sleep(0.1)  # Let threads run
        
        core.stop()
        assert core.is_running is False
    
    def test_coordinate_transformation(self, core):
        """Test coordinate space transformations"""
        # Set up transforms
        core.coordinate_transforms[CoordinateSpace.WORLD.value].position = np.array([0, 0, 0])
        core.coordinate_transforms[CoordinateSpace.DEVICE.value].position = np.array([1, 2, 3])
        
        # Transform point from world to device space
        world_point = np.array([5, 5, 5])
        device_point = core.transform_point(world_point, CoordinateSpace.WORLD, CoordinateSpace.DEVICE)
        
        assert device_point is not None
        assert device_point.shape == (3,)
    
    def test_project_to_screen(self, core):
        """Test world to screen projection"""
        device = MockARVRDevice("test_device", DeviceType.HMD_AR)
        device.render_target = RenderTarget(
            width=1920,
            height=1080,
            fov=90
        )
        core.add_device(device)
        
        # Point in front of camera
        world_point = np.array([0, 0, -10])
        screen_pos = core.project_to_screen(world_point)
        
        assert screen_pos is not None
        assert 0 <= screen_pos[0] < 1920
        assert 0 <= screen_pos[1] < 1080
        
        # Point behind camera
        world_point = np.array([0, 0, 10])
        screen_pos = core.project_to_screen(world_point)
        assert screen_pos is None
    
    def test_render_resolution(self, core):
        """Test setting render resolution"""
        core.set_render_resolution(3840, 2160)
        assert core.render_resolution == (3840, 2160)
        assert core.frame_buffer is None  # Should be recreated
    
    def test_configuration(self, core):
        """Test configuration updates"""
        config = {
            'enable_reprojection': False,
            'enable_foveated_rendering': True,
            'fps_target': 120
        }
        
        core.set_config(config)
        assert core.config['enable_reprojection'] is False
        assert core.config['enable_foveated_rendering'] is True
        assert core.fps_target == 90  # fps_target is separate attribute
    
    def test_metrics(self, core):
        """Test performance metrics"""
        metrics = core.get_metrics()
        
        assert 'fps' in metrics
        assert 'frame_time' in metrics
        assert 'tracking_latency' in metrics
        assert 'render_latency' in metrics
        assert 'total_latency' in metrics
    
    def test_callbacks(self, core):
        """Test callback registration and firing"""
        mock_callback = Mock()
        core.set_callback('on_device_connected', mock_callback)
        
        # Add device should trigger callback
        device = MockARVRDevice("test_device", DeviceType.HMD_VR)
        core.add_device(device)
        
        mock_callback.assert_called_once_with(device)
    
    def test_capture_frame(self, core):
        """Test frame capture"""
        # No frame buffer initially
        frame = core.capture_frame()
        assert frame is None
        
        # Create frame buffer
        core.frame_buffer = np.zeros((1080, 1920, 4), dtype=np.uint8)
        frame = core.capture_frame()
        assert frame is not None
        assert frame.shape == (1080, 1920, 4)
    
    def test_reset_tracking(self, core):
        """Test tracking reset"""
        # Set some transforms
        core.coordinate_transforms[CoordinateSpace.DEVICE.value].position = np.array([1, 2, 3])
        
        core.reset_tracking()
        
        # All transforms should be reset
        for transform in core.coordinate_transforms.values():
            assert np.array_equal(transform.position, np.zeros(3))
            assert np.array_equal(transform.rotation, np.array([0, 0, 0, 1]))
            assert np.array_equal(transform.scale, np.ones(3))


class TestDeviceType:
    """Test DeviceType enumeration"""
    
    def test_device_types(self):
        """Test device type values"""
        assert DeviceType.HMD_VR.value == "hmd_vr"
        assert DeviceType.HMD_AR.value == "hmd_ar"
        assert DeviceType.HOLOGRAPHIC.value == "holographic"
        assert DeviceType.PROJECTOR.value == "projector"
        assert DeviceType.MOBILE_AR.value == "mobile_ar"


class TestRenderMode:
    """Test RenderMode enumeration"""
    
    def test_render_modes(self):
        """Test render mode values"""
        assert RenderMode.STEREO_VR.value == "stereo_vr"
        assert RenderMode.MONO_AR.value == "mono_ar"
        assert RenderMode.HOLOGRAPHIC_3D.value == "holographic_3d"
        assert RenderMode.PROJECTION_MAPPING.value == "projection_mapping"


class TestCoordinateSpace:
    """Test CoordinateSpace enumeration"""
    
    def test_coordinate_spaces(self):
        """Test coordinate space values"""
        assert CoordinateSpace.WORLD.value == "world"
        assert CoordinateSpace.DEVICE.value == "device"
        assert CoordinateSpace.USER.value == "user"
        assert CoordinateSpace.SCREEN.value == "screen"