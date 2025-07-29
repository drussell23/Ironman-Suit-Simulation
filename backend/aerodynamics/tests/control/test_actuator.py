"""
Unit tests for the Actuator module
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import time

from aerodynamics.control.actuator import (
    Actuator, 
    ThrusterActuator, 
    ControlSurfaceActuator,
    ActuatorGroup,
    ActuatorError
)


class TestActuator:
    """Test base Actuator functionality"""
    
    def test_base_initialization(self):
        """Test base actuator initialization"""
        actuator = Actuator(
            name="test_actuator",
            max_output=100.0,
            min_output=-100.0,
            response_time=0.1
        )
        
        assert actuator.name == "test_actuator"
        assert actuator.max_output == 100.0
        assert actuator.min_output == -100.0
        assert actuator.response_time == 0.1
        assert actuator.current_output == 0.0
        assert actuator.is_active is True
    
    def test_command_saturation(self):
        """Test command saturation within limits"""
        actuator = Actuator(max_output=50.0, min_output=-50.0)
        
        # Test upper saturation
        actuator.set_command(100.0)
        assert actuator.get_output() == 50.0
        
        # Test lower saturation
        actuator.set_command(-100.0)
        assert actuator.get_output() == -50.0
        
        # Test within limits
        actuator.set_command(25.0)
        assert actuator.get_output() == 25.0
    
    def test_actuator_dynamics(self):
        """Test actuator response dynamics"""
        actuator = Actuator(response_time=0.5)
        
        # Step command
        actuator.set_command(100.0)
        
        # Immediate response should be less than command
        dt = 0.1
        output1 = actuator.update(dt)
        assert output1 < 100.0
        assert output1 > 0.0
        
        # After several updates, should approach command
        for _ in range(10):
            output = actuator.update(dt)
        
        assert abs(output - 100.0) < 5.0  # Within 5% of command
    
    def test_actuator_failure(self):
        """Test actuator failure modes"""
        actuator = Actuator(name="test")
        
        # Normal operation
        actuator.set_command(50.0)
        assert actuator.get_output() == 50.0
        
        # Simulate failure
        actuator.fail()
        assert not actuator.is_active
        assert actuator.get_output() == 0.0
        
        # Commands should not affect failed actuator
        actuator.set_command(100.0)
        assert actuator.get_output() == 0.0
        
        # Reset actuator
        actuator.reset()
        assert actuator.is_active
        actuator.set_command(50.0)
        assert actuator.get_output() == 50.0


class TestThrusterActuator:
    """Test ThrusterActuator functionality"""
    
    def test_thruster_initialization(self):
        """Test thruster actuator initialization"""
        thruster = ThrusterActuator(
            name="main_thruster",
            max_thrust=5000.0,
            min_thrust=0.0,
            response_time=0.05,
            efficiency=0.85,
            fuel_consumption_rate=0.1
        )
        
        assert thruster.name == "main_thruster"
        assert thruster.max_thrust == 5000.0
        assert thruster.min_thrust == 0.0
        assert thruster.efficiency == 0.85
        assert thruster.fuel_consumption_rate == 0.1
    
    def test_thrust_output(self):
        """Test thrust output calculation"""
        thruster = ThrusterActuator(
            max_thrust=1000.0,
            efficiency=0.9
        )
        
        # Set thrust command (0-1 normalized)
        thruster.set_command(0.5)
        thrust = thruster.get_thrust()
        
        assert thrust == pytest.approx(500.0 * 0.9)  # 50% * max * efficiency
    
    def test_fuel_consumption(self):
        """Test fuel consumption calculation"""
        thruster = ThrusterActuator(
            max_thrust=1000.0,
            fuel_consumption_rate=0.1  # kg/s per N
        )
        
        thruster.set_command(0.8)  # 80% thrust
        thrust = thruster.get_thrust()
        fuel_rate = thruster.get_fuel_consumption_rate()
        
        expected_rate = thrust * 0.1
        assert fuel_rate == pytest.approx(expected_rate)
    
    def test_thrust_vectoring(self):
        """Test thrust vectoring capability"""
        if hasattr(ThrusterActuator, 'set_vector_angle'):
            thruster = ThrusterActuator(
                max_thrust=1000.0,
                max_vector_angle=np.deg2rad(15)  # 15 degrees
            )
            
            # Set thrust vector angles
            thruster.set_vector_angle(pitch=np.deg2rad(10), yaw=np.deg2rad(5))
            
            # Get thrust vector
            thrust_vector = thruster.get_thrust_vector()
            
            # Check vector magnitude and direction
            thrust_mag = np.linalg.norm(thrust_vector)
            assert thrust_mag > 0
            
            # Check angles are within limits
            assert abs(thruster.pitch_angle) <= thruster.max_vector_angle
            assert abs(thruster.yaw_angle) <= thruster.max_vector_angle


class TestControlSurfaceActuator:
    """Test ControlSurfaceActuator functionality"""
    
    def test_control_surface_initialization(self):
        """Test control surface actuator initialization"""
        surface = ControlSurfaceActuator(
            name="aileron_left",
            max_deflection=np.deg2rad(30),
            min_deflection=np.deg2rad(-30),
            response_time=0.1,
            area=0.5
        )
        
        assert surface.name == "aileron_left"
        assert surface.max_deflection == np.deg2rad(30)
        assert surface.min_deflection == np.deg2rad(-30)
        assert surface.area == 0.5
    
    def test_deflection_limits(self):
        """Test deflection angle limits"""
        surface = ControlSurfaceActuator(
            max_deflection=np.deg2rad(25),
            min_deflection=np.deg2rad(-25)
        )
        
        # Test saturation
        surface.set_deflection(np.deg2rad(40))
        assert surface.get_deflection() == np.deg2rad(25)
        
        surface.set_deflection(np.deg2rad(-40))
        assert surface.get_deflection() == np.deg2rad(-25)
    
    def test_aerodynamic_coefficients(self):
        """Test aerodynamic coefficient calculation"""
        surface = ControlSurfaceActuator(
            area=0.5,
            effectiveness=0.8
        )
        
        # Set deflection
        deflection = np.deg2rad(10)
        surface.set_deflection(deflection)
        
        # Get coefficients
        if hasattr(surface, 'get_force_coefficient'):
            coeff = surface.get_force_coefficient(
                airspeed=50.0,
                air_density=1.225
            )
            
            assert coeff != 0.0
            assert abs(coeff) < 1.0  # Reasonable coefficient range


class TestActuatorGroup:
    """Test ActuatorGroup functionality"""
    
    def test_group_initialization(self):
        """Test actuator group initialization"""
        group = ActuatorGroup(name="thrusters")
        
        assert group.name == "thrusters"
        assert len(group.actuators) == 0
    
    def test_add_remove_actuators(self):
        """Test adding and removing actuators from group"""
        group = ActuatorGroup()
        
        # Create actuators
        actuator1 = Actuator(name="act1")
        actuator2 = Actuator(name="act2")
        
        # Add actuators
        group.add_actuator(actuator1)
        group.add_actuator(actuator2)
        
        assert len(group.actuators) == 2
        assert "act1" in group.get_actuator_names()
        assert "act2" in group.get_actuator_names()
        
        # Remove actuator
        group.remove_actuator("act1")
        assert len(group.actuators) == 1
        assert "act1" not in group.get_actuator_names()
    
    def test_group_commands(self):
        """Test sending commands to actuator group"""
        group = ActuatorGroup()
        
        # Add multiple actuators
        for i in range(4):
            actuator = Actuator(name=f"act{i}", max_output=100.0)
            group.add_actuator(actuator)
        
        # Set group command
        group.set_group_command(50.0)
        
        # Check all actuators received command
        for actuator in group.actuators.values():
            assert actuator.get_output() == 50.0
    
    def test_differential_commands(self):
        """Test differential commands to actuators"""
        group = ActuatorGroup()
        
        # Add actuators
        left = Actuator(name="left")
        right = Actuator(name="right")
        group.add_actuator(left)
        group.add_actuator(right)
        
        # Set differential commands
        commands = {"left": 75.0, "right": 25.0}
        group.set_commands(commands)
        
        assert group.get_actuator("left").get_output() == 75.0
        assert group.get_actuator("right").get_output() == 25.0
    
    def test_group_failure_handling(self):
        """Test handling of actuator failures in group"""
        group = ActuatorGroup()
        
        # Add actuators
        for i in range(4):
            actuator = ThrusterActuator(name=f"thruster{i}", max_thrust=1000.0)
            group.add_actuator(actuator)
        
        # Fail one actuator
        group.get_actuator("thruster1").fail()
        
        # Get total output
        if hasattr(group, 'get_total_thrust'):
            group.set_group_command(1.0)  # Full thrust
            total_thrust = group.get_total_thrust()
            
            # Should be 3/4 of maximum (one failed)
            assert total_thrust == pytest.approx(3000.0, rel=0.1)
    
    def test_actuator_allocation(self):
        """Test actuator allocation for desired forces/moments"""
        if hasattr(ActuatorGroup, 'allocate_control'):
            group = ActuatorGroup()
            
            # Add thrusters in different positions
            positions = [
                np.array([1.0, 0.0, 0.0]),
                np.array([-1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, -1.0, 0.0])
            ]
            
            for i, pos in enumerate(positions):
                thruster = ThrusterActuator(
                    name=f"thruster{i}",
                    max_thrust=1000.0,
                    position=pos
                )
                group.add_actuator(thruster)
            
            # Request force and moment
            desired_force = np.array([100.0, 0.0, 0.0])
            desired_moment = np.array([0.0, 0.0, 50.0])
            
            # Allocate to actuators
            commands = group.allocate_control(desired_force, desired_moment)
            
            assert len(commands) == 4
            assert all(0 <= cmd <= 1.0 for cmd in commands.values())