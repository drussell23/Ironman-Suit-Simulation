"""
Tests for Virtual Interface system.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch
from backend.ar_vr_integration.virtual_interface import (
    VirtualInterface, InterfaceType, InteractionMode, VirtualButton,
    VirtualSlider, VirtualPanel, VirtualObject
)


class TestVirtualInterface:
    """Test cases for VirtualInterface class"""
    
    @pytest.fixture
    def vi(self):
        """Create virtual interface instance for testing"""
        return VirtualInterface()
    
    def test_initialization(self, vi):
        """Test virtual interface initialization"""
        assert len(vi.interfaces) > 0  # Default interfaces created
        assert vi.active_interface is None
        assert vi.interaction_mode == InteractionMode.HYBRID
        assert vi.is_active is False
    
    def test_default_interfaces(self, vi):
        """Test default interface creation"""
        assert "main_control" in vi.interfaces
        assert "weapon_systems" in vi.interfaces
        assert "communications" in vi.interfaces
        assert "diagnostics" in vi.interfaces
    
    def test_start_stop(self, vi):
        """Test starting and stopping interface"""
        vi.start()
        assert vi.is_active is True
        assert vi.update_thread is not None
        assert vi.update_thread.is_alive()
        
        vi.stop()
        assert vi.is_active is False
    
    def test_show_hide_interface(self, vi):
        """Test showing and hiding interfaces"""
        vi.show_interface("main_control")
        assert vi.active_interface == "main_control"
        assert vi.interfaces["main_control"].visible is True
        
        vi.hide_interface("main_control")
        assert vi.interfaces["main_control"].visible is False
        assert vi.active_interface is None
    
    def test_add_remove_interface(self, vi):
        """Test adding and removing interfaces"""
        panel = VirtualPanel(
            id="test_panel",
            name="Test Panel",
            position=np.array([0, 1, 1]),
            rotation=np.array([0, 0, 0]),
            scale=np.array([1, 1, 1])
        )
        
        vi.add_interface("test_panel", panel)
        assert "test_panel" in vi.interfaces
        
        vi.remove_interface("test_panel")
        assert "test_panel" not in vi.interfaces
    
    def test_interface_limit(self, vi):
        """Test interface limit enforcement"""
        vi.config['max_interfaces'] = 5
        
        # Add interfaces up to limit
        for i in range(10):
            panel = VirtualPanel(
                id=f"panel_{i}",
                name=f"Panel {i}",
                position=np.array([0, 1, 1]),
                rotation=np.array([0, 0, 0]),
                scale=np.array([1, 1, 1])
            )
            vi.add_interface(f"panel_{i}", panel)
        
        assert len(vi.interfaces) <= vi.config['max_interfaces']
    
    def test_virtual_button(self):
        """Test virtual button creation"""
        button = VirtualButton(
            id="test_button",
            name="Test Button",
            position=np.array([0, 0, 0]),
            rotation=np.array([0, 0, 0]),
            scale=np.array([1, 1, 1]),
            label="TEST",
            callback=lambda: None
        )
        
        assert button.id == "test_button"
        assert button.label == "TEST"
        assert button.pressed is False
        assert button.hover is False
    
    def test_virtual_slider(self):
        """Test virtual slider creation"""
        slider = VirtualSlider(
            id="test_slider",
            name="Test Slider",
            position=np.array([0, 0, 0]),
            rotation=np.array([0, 0, 0]),
            scale=np.array([1, 1, 1]),
            min_value=0,
            max_value=100,
            current_value=50
        )
        
        assert slider.id == "test_slider"
        assert slider.min_value == 0
        assert slider.max_value == 100
        assert slider.current_value == 50
    
    def test_hand_position_update(self, vi):
        """Test updating hand positions"""
        left_pos = np.array([0.1, 0.2, 0.3])
        right_pos = np.array([0.4, 0.5, 0.6])
        
        vi.update_hand_position('left', left_pos)
        vi.update_hand_position('right', right_pos)
        
        assert np.array_equal(vi.hand_positions['left'], left_pos)
        assert np.array_equal(vi.hand_positions['right'], right_pos)
    
    def test_user_transform_update(self, vi):
        """Test updating user position and orientation"""
        position = np.array([1, 2, 3])
        orientation = np.array([0, 1, 0])
        
        vi.update_user_transform(position, orientation)
        
        assert np.array_equal(vi.user_position, position)
        assert np.array_equal(vi.user_orientation, orientation)
    
    def test_gesture_registration(self, vi):
        """Test registering gestures"""
        gesture_data = {
            'type': 'pinch',
            'hand': 'right',
            'position': [0.5, 0.5, 0.5],
            'timestamp': time.time()
        }
        
        vi.register_gesture(gesture_data)
        assert len(vi.gesture_buffer) > 0
    
    def test_callback_registration(self, vi):
        """Test callback registration"""
        mock_callback = Mock()
        
        vi.set_callback('on_select', mock_callback)
        assert vi.callbacks['on_select'] == mock_callback
    
    def test_get_object(self, vi):
        """Test getting objects by ID"""
        # Main control panel should have objects
        vi.show_interface("main_control")
        
        # Get power slider
        obj = vi.get_object("power_slider")
        assert obj is not None
        assert isinstance(obj, VirtualSlider)
    
    def test_update_object(self, vi):
        """Test updating object properties"""
        vi.show_interface("main_control")
        
        # Update power slider
        vi.update_object("power_slider", current_value=80, min_value=10)
        
        obj = vi.get_object("power_slider")
        assert obj.current_value == 80
        assert obj.min_value == 10
    
    def test_custom_interface_creation(self, vi):
        """Test creating custom interface from config"""
        config = {
            'name': 'Custom Interface',
            'position': [0, 1.5, 1],
            'width': 0.8,
            'height': 0.6,
            'elements': [
                {
                    'type': 'button',
                    'id': 'custom_button',
                    'name': 'Custom Button',
                    'position': [0, 0, 0],
                    'label': 'CUSTOM'
                },
                {
                    'type': 'slider',
                    'id': 'custom_slider',
                    'name': 'Custom Slider',
                    'position': [0, -0.1, 0],
                    'min': 0,
                    'max': 10,
                    'value': 5
                }
            ]
        }
        
        panel = vi.create_custom_interface('custom_interface', config)
        assert panel is not None
        assert len(panel.elements) == 2
        assert 'custom_interface' in vi.interfaces
    
    def test_interface_state(self, vi):
        """Test getting interface state"""
        vi.show_interface("main_control")
        
        state = vi.get_interface_state()
        assert state['active_interface'] == "main_control"
        assert 'interfaces' in state
        assert 'main_control' in state['interfaces']
        assert state['interfaces']['main_control']['visible'] is True
    
    def test_interface_orientation(self, vi):
        """Test interface orientation to user"""
        vi.user_position = np.array([0, 0, -2])
        
        # Create interface in front
        panel = VirtualPanel(
            id="front_panel",
            name="Front Panel",
            position=np.array([0, 1, 0]),
            rotation=np.array([0, 0, 0]),
            scale=np.array([1, 1, 1])
        )
        
        vi.add_interface("front_panel", panel)
        vi.show_interface("front_panel")
        
        # Start interface to trigger orientation updates
        vi.start()
        time.sleep(0.1)
        vi.stop()
        
        # Panel should have rotated to face user
        # (actual rotation calculation depends on implementation)
        assert panel.rotation[1] != 0  # Y rotation should change


class TestInteractionMode:
    """Test interaction mode enumeration"""
    
    def test_mode_values(self):
        """Test interaction mode values"""
        assert InteractionMode.GESTURE.value == "gesture"
        assert InteractionMode.VOICE.value == "voice"
        assert InteractionMode.GAZE.value == "gaze"
        assert InteractionMode.HYBRID.value == "hybrid"


class TestInterfaceType:
    """Test interface type enumeration"""
    
    def test_type_values(self):
        """Test interface type values"""
        assert InterfaceType.CONTROL_PANEL.value == "control_panel"
        assert InterfaceType.HOLOGRAPHIC_MAP.value == "holographic_map"
        assert InterfaceType.DATA_VISUALIZATION.value == "data_visualization"
        assert InterfaceType.WEAPON_SYSTEMS.value == "weapon_systems"
        assert InterfaceType.COMMUNICATIONS.value == "communications"
        assert InterfaceType.DIAGNOSTICS.value == "diagnostics"