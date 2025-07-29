"""
Unit tests for the Controller module
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from aerodynamics.control.controller import Controller, ControllerError


class TestController:
    """Test the PD Controller functionality"""
    
    @pytest.fixture
    def controller(self):
        """Create a controller instance with default parameters"""
        return Controller(
            mass=100.0,
            inertia=np.diag([10.0, 10.0, 10.0]),
            kp_trans=2.0,
            kd_trans=0.5,
            kp_rot=1.5,
            kd_rot=0.3
        )
    
    def test_initialization(self):
        """Test controller initialization"""
        controller = Controller()
        
        assert controller.mass == 1.0
        assert np.array_equal(controller.inertia, np.eye(3))
        assert controller.kp_trans == 1.0
        assert controller.kd_trans == 0.0
        assert controller.kp_rot == 1.0
        assert controller.kd_rot == 0.0
        assert np.array_equal(controller._prev_vel_error, np.zeros(3))
        assert np.array_equal(controller._prev_omega_error, np.zeros(3))
    
    def test_custom_initialization(self):
        """Test controller with custom parameters"""
        mass = 150.0
        inertia = np.diag([20.0, 25.0, 30.0])
        
        controller = Controller(
            mass=mass,
            inertia=inertia,
            kp_trans=3.0,
            kd_trans=1.0,
            kp_rot=2.0,
            kd_rot=0.5
        )
        
        assert controller.mass == mass
        assert np.array_equal(controller.inertia, inertia)
        assert controller.kp_trans == 3.0
        assert controller.kd_trans == 1.0
        assert controller.kp_rot == 2.0
        assert controller.kd_rot == 0.5
    
    def test_compute_control_translation(self, controller):
        """Test translation control computation"""
        state = {
            "velocity": np.array([10.0, 5.0, 0.0]),
            "omega": np.array([0.0, 0.0, 0.0])
        }
        
        reference = {
            "velocity": np.array([15.0, 5.0, 2.0]),
            "omega": np.array([0.0, 0.0, 0.0])
        }
        
        dt = 0.01
        
        control = controller.compute_control(state, reference, dt)
        
        assert "force" in control
        assert "moment" in control
        assert control["force"].shape == (3,)
        assert control["moment"].shape == (3,)
        
        # Check force calculation
        vel_error = reference["velocity"] - state["velocity"]
        expected_force = controller.mass * controller.kp_trans * vel_error
        
        # First call should have zero derivative term
        np.testing.assert_allclose(control["force"], expected_force, rtol=1e-10)
    
    def test_compute_control_rotation(self, controller):
        """Test rotation control computation"""
        state = {
            "velocity": np.array([0.0, 0.0, 0.0]),
            "omega": np.array([0.1, -0.05, 0.0])
        }
        
        reference = {
            "velocity": np.array([0.0, 0.0, 0.0]),
            "omega": np.array([0.0, 0.0, 0.1])
        }
        
        dt = 0.01
        
        control = controller.compute_control(state, reference, dt)
        
        # Check moment calculation
        omega_error = reference["omega"] - state["omega"]
        expected_moment = controller.inertia @ (controller.kp_rot * omega_error)
        
        np.testing.assert_allclose(control["moment"], expected_moment, rtol=1e-10)
    
    def test_compute_control_with_derivative(self, controller):
        """Test control with derivative term"""
        state1 = {
            "velocity": np.array([10.0, 0.0, 0.0]),
            "omega": np.array([0.0, 0.0, 0.0])
        }
        
        reference = {
            "velocity": np.array([20.0, 0.0, 0.0]),
            "omega": np.array([0.0, 0.0, 0.1])
        }
        
        dt = 0.01
        
        # First control computation
        control1 = controller.compute_control(state1, reference, dt)
        
        # Second state with changed velocity
        state2 = {
            "velocity": np.array([12.0, 0.0, 0.0]),
            "omega": np.array([0.0, 0.0, 0.02])
        }
        
        # Second control computation
        control2 = controller.compute_control(state2, reference, dt)
        
        # Now derivative term should be non-zero
        assert not np.array_equal(control1["force"], control2["force"])
        assert not np.array_equal(control1["moment"], control2["moment"])
    
    def test_control_saturation(self, controller):
        """Test control output saturation"""
        # Set max force/moment if implemented
        if hasattr(controller, 'max_force'):
            state = {
                "velocity": np.array([0.0, 0.0, 0.0]),
                "omega": np.array([0.0, 0.0, 0.0])
            }
            
            # Very large reference to trigger saturation
            reference = {
                "velocity": np.array([1000.0, 1000.0, 1000.0]),
                "omega": np.array([100.0, 100.0, 100.0])
            }
            
            control = controller.compute_control(state, reference, 0.01)
            
            # Check saturation
            force_mag = np.linalg.norm(control["force"])
            assert force_mag <= controller.max_force
    
    def test_zero_dt_handling(self, controller):
        """Test handling of zero time step"""
        state = {
            "velocity": np.array([10.0, 0.0, 0.0]),
            "omega": np.array([0.0, 0.0, 0.0])
        }
        
        reference = {
            "velocity": np.array([15.0, 0.0, 0.0]),
            "omega": np.array([0.0, 0.0, 0.0])
        }
        
        # Should handle dt=0 without division by zero
        control = controller.compute_control(state, reference, 0.0)
        
        assert not np.any(np.isnan(control["force"]))
        assert not np.any(np.isnan(control["moment"]))
    
    def test_missing_state_keys(self, controller):
        """Test error handling for missing state keys"""
        incomplete_state = {
            "velocity": np.array([10.0, 0.0, 0.0])
            # Missing omega
        }
        
        reference = {
            "velocity": np.array([15.0, 0.0, 0.0]),
            "omega": np.array([0.0, 0.0, 0.0])
        }
        
        with pytest.raises((KeyError, ControllerError)):
            controller.compute_control(incomplete_state, reference, 0.01)
    
    def test_reset_controller(self, controller):
        """Test controller reset functionality"""
        # Perform some control computations
        state = {
            "velocity": np.array([10.0, 5.0, 0.0]),
            "omega": np.array([0.1, 0.0, 0.0])
        }
        
        reference = {
            "velocity": np.array([15.0, 5.0, 0.0]),
            "omega": np.array([0.0, 0.0, 0.0])
        }
        
        controller.compute_control(state, reference, 0.01)
        
        # Reset if method exists
        if hasattr(controller, 'reset'):
            controller.reset()
            assert np.array_equal(controller._prev_vel_error, np.zeros(3))
            assert np.array_equal(controller._prev_omega_error, np.zeros(3))
    
    def test_gain_scheduling(self, controller):
        """Test gain scheduling based on flight conditions"""
        if hasattr(controller, 'update_gains'):
            # High speed condition
            high_speed_state = {
                "velocity": np.array([200.0, 0.0, 0.0]),
                "altitude": 5000.0
            }
            
            controller.update_gains(high_speed_state)
            
            # Gains might be adjusted for high speed
            assert controller.kp_trans > 0
            assert controller.kd_trans >= 0