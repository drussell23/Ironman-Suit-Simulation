"""
Unit tests for the Autopilot module
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from aerodynamics.control.autopilot import Autopilot, ControllerError


class TestAutopilot:
    """Test the Autopilot functionality"""
    
    @pytest.fixture
    def autopilot(self):
        """Create an autopilot instance"""
        controller_mock = Mock()
        controller_mock.compute_control.return_value = {
            "force": np.array([100.0, 0.0, 0.0]),
            "moment": np.array([0.0, 0.0, 10.0])
        }
        
        return Autopilot(controller=controller_mock)
    
    def test_initialization(self):
        """Test autopilot initialization"""
        autopilot = Autopilot()
        
        assert autopilot.controller is not None
        assert autopilot.is_engaged is False
        assert autopilot.mode == "manual"
        assert autopilot.waypoints == []
        assert autopilot.current_waypoint_idx == 0
    
    def test_engage_disengage(self, autopilot):
        """Test autopilot engagement and disengagement"""
        # Initially disengaged
        assert not autopilot.is_engaged
        
        # Engage
        autopilot.engage()
        assert autopilot.is_engaged
        
        # Disengage
        autopilot.disengage()
        assert not autopilot.is_engaged
    
    def test_set_mode(self, autopilot):
        """Test setting autopilot modes"""
        valid_modes = ["manual", "altitude_hold", "heading_hold", "waypoint", "hover"]
        
        for mode in valid_modes:
            autopilot.set_mode(mode)
            assert autopilot.mode == mode
        
        # Test invalid mode
        with pytest.raises(ValueError):
            autopilot.set_mode("invalid_mode")
    
    def test_altitude_hold_mode(self, autopilot):
        """Test altitude hold mode"""
        autopilot.engage()
        autopilot.set_mode("altitude_hold")
        autopilot.set_target_altitude(1000.0)
        
        state = {
            "position": np.array([0.0, 0.0, 800.0]),
            "velocity": np.array([50.0, 0.0, 5.0]),
            "orientation": np.array([1.0, 0.0, 0.0, 0.0]),  # quaternion
            "omega": np.array([0.0, 0.0, 0.0])
        }
        
        control = autopilot.compute_command(state, dt=0.01)
        
        assert "force" in control
        assert "moment" in control
        # Should command upward force to reach target altitude
        assert control["force"][2] > 0  # Positive Z force
    
    def test_heading_hold_mode(self, autopilot):
        """Test heading hold mode"""
        autopilot.engage()
        autopilot.set_mode("heading_hold")
        autopilot.set_target_heading(np.pi/2)  # 90 degrees
        
        state = {
            "position": np.array([0.0, 0.0, 1000.0]),
            "velocity": np.array([50.0, 0.0, 0.0]),
            "orientation": np.array([1.0, 0.0, 0.0, 0.0]),  # facing forward
            "omega": np.array([0.0, 0.0, 0.0])
        }
        
        control = autopilot.compute_command(state, dt=0.01)
        
        # Should command yaw moment to turn right
        assert control["moment"][2] != 0
    
    def test_waypoint_navigation(self, autopilot):
        """Test waypoint navigation mode"""
        waypoints = [
            np.array([100.0, 0.0, 1000.0]),
            np.array([100.0, 100.0, 1000.0]),
            np.array([0.0, 100.0, 1000.0])
        ]
        
        autopilot.set_waypoints(waypoints)
        autopilot.engage()
        autopilot.set_mode("waypoint")
        
        # Start position
        state = {
            "position": np.array([0.0, 0.0, 1000.0]),
            "velocity": np.array([0.0, 0.0, 0.0]),
            "orientation": np.array([1.0, 0.0, 0.0, 0.0]),
            "omega": np.array([0.0, 0.0, 0.0])
        }
        
        control = autopilot.compute_command(state, dt=0.01)
        
        # Should command movement toward first waypoint
        assert control["force"][0] > 0  # Move in +X direction
        
        # Check waypoint reached detection
        if hasattr(autopilot, 'check_waypoint_reached'):
            state["position"] = waypoints[0].copy()
            reached = autopilot.check_waypoint_reached(state)
            assert reached
            assert autopilot.current_waypoint_idx == 1
    
    def test_hover_mode(self, autopilot):
        """Test hover mode"""
        autopilot.engage()
        autopilot.set_mode("hover")
        
        hover_position = np.array([50.0, 50.0, 500.0])
        autopilot.set_hover_position(hover_position)
        
        # Current state with some drift
        state = {
            "position": np.array([52.0, 48.0, 498.0]),
            "velocity": np.array([1.0, -0.5, -0.2]),
            "orientation": np.array([1.0, 0.0, 0.0, 0.0]),
            "omega": np.array([0.0, 0.0, 0.0])
        }
        
        control = autopilot.compute_command(state, dt=0.01)
        
        # Should command forces to return to hover position
        # and counteract velocity
        assert control["force"][0] < 0  # Counteract positive X drift
        assert control["force"][1] > 0  # Move back to Y position
        assert control["force"][2] > 0  # Move back to Z position
    
    def test_command_limits(self, autopilot):
        """Test command limiting for safety"""
        if hasattr(autopilot, 'max_force') and hasattr(autopilot, 'max_moment'):
            autopilot.max_force = 1000.0
            autopilot.max_moment = 100.0
            
            # Set controller to return excessive commands
            autopilot.controller.compute_control.return_value = {
                "force": np.array([2000.0, 2000.0, 2000.0]),
                "moment": np.array([200.0, 200.0, 200.0])
            }
            
            state = {
                "position": np.array([0.0, 0.0, 1000.0]),
                "velocity": np.array([0.0, 0.0, 0.0]),
                "orientation": np.array([1.0, 0.0, 0.0, 0.0]),
                "omega": np.array([0.0, 0.0, 0.0])
            }
            
            control = autopilot.compute_command(state, dt=0.01)
            
            # Check limits are enforced
            force_mag = np.linalg.norm(control["force"])
            moment_mag = np.linalg.norm(control["moment"])
            
            assert force_mag <= autopilot.max_force
            assert moment_mag <= autopilot.max_moment
    
    def test_manual_override(self, autopilot):
        """Test manual override functionality"""
        autopilot.engage()
        autopilot.set_mode("altitude_hold")
        
        state = {
            "position": np.array([0.0, 0.0, 1000.0]),
            "velocity": np.array([50.0, 0.0, 0.0]),
            "orientation": np.array([1.0, 0.0, 0.0, 0.0]),
            "omega": np.array([0.0, 0.0, 0.0])
        }
        
        manual_input = {
            "force": np.array([50.0, 0.0, 0.0]),
            "moment": np.array([0.0, 10.0, 0.0])
        }
        
        # Apply manual override
        if hasattr(autopilot, 'apply_manual_override'):
            control = autopilot.compute_command(state, dt=0.01, manual_override=manual_input)
            
            # Manual input should influence the output
            assert not np.array_equal(control["force"], autopilot.controller.compute_control.return_value["force"])
    
    def test_emergency_stop(self, autopilot):
        """Test emergency stop functionality"""
        if hasattr(autopilot, 'emergency_stop'):
            autopilot.engage()
            autopilot.set_mode("waypoint")
            
            # Trigger emergency stop
            autopilot.emergency_stop()
            
            assert not autopilot.is_engaged
            assert autopilot.mode == "manual"
            
            # Commands should be zero or minimal
            state = {
                "position": np.array([0.0, 0.0, 1000.0]),
                "velocity": np.array([50.0, 0.0, 0.0]),
                "orientation": np.array([1.0, 0.0, 0.0, 0.0]),
                "omega": np.array([0.0, 0.0, 0.0])
            }
            
            control = autopilot.compute_command(state, dt=0.01)
            
            # Should return safe default commands
            assert np.linalg.norm(control["force"]) < 100.0
    
    def test_state_validation(self, autopilot):
        """Test state input validation"""
        autopilot.engage()
        
        # Missing required state fields
        invalid_state = {
            "position": np.array([0.0, 0.0, 1000.0])
            # Missing velocity, orientation, omega
        }
        
        with pytest.raises((KeyError, ControllerError)):
            autopilot.compute_command(invalid_state, dt=0.01)
        
        # Invalid array shapes
        bad_shape_state = {
            "position": np.array([0.0, 0.0]),  # Should be 3D
            "velocity": np.array([0.0, 0.0, 0.0]),
            "orientation": np.array([1.0, 0.0, 0.0, 0.0]),
            "omega": np.array([0.0, 0.0, 0.0])
        }
        
        with pytest.raises((ValueError, ControllerError)):
            autopilot.compute_command(bad_shape_state, dt=0.01)