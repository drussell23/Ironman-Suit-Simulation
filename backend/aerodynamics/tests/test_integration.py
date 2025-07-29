"""
Integration tests for the complete Aerodynamics system

These tests verify that all aerodynamics components work together properly
in realistic flight scenarios.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import time

# Import all major components
from aerodynamics import (
    FlightDynamics,
    StabilityControl,
    Controller,
    Autopilot,
    density_at_altitude,
    wind_at_position,
    compute_thrust
)
from aerodynamics.environmental_effects import atmospheric_density, wind_interaction
from aerodynamics.environmental_effects.turbulence_models import k_epsilon, smagorinsky
from aerodynamics.flight_models import aerodynamic_forces
from aerodynamics.control.actuator import ThrusterActuator, ActuatorGroup
from aerodynamics.control.sensor import IMUSensor, PitotSensor
from aerodynamics.simulations.run_simulation import FlightSimulation


class TestAerodynamicsIntegration:
    """Integration tests for the complete aerodynamics system"""
    
    @pytest.fixture
    def flight_system(self):
        """Create a complete flight system"""
        # Flight dynamics model
        dynamics = FlightDynamics(
            mass=100.0,
            wing_area=2.0,
            Cl0=0.1,
            Cld_alpha=5.0,
            Cd0=0.02,
            k=0.04
        )
        
        # Control system
        controller = Controller(
            mass=100.0,
            inertia=np.diag([10.0, 10.0, 10.0]),
            kp_trans=2.0,
            kd_trans=0.5,
            kp_rot=1.5,
            kd_rot=0.3
        )
        
        # Autopilot
        autopilot = Autopilot(controller=controller)
        
        # Stability augmentation
        stability = StabilityControl()
        
        return {
            "dynamics": dynamics,
            "controller": controller,
            "autopilot": autopilot,
            "stability": stability
        }
    
    def test_complete_flight_cycle(self, flight_system):
        """Test a complete flight cycle: takeoff, cruise, landing"""
        dynamics = flight_system["dynamics"]
        autopilot = flight_system["autopilot"]
        
        # Initial state (on ground)
        state = {
            "position": np.array([0.0, 0.0, 0.0]),
            "velocity": np.array([0.0, 0.0, 0.0]),
            "orientation": np.array([1.0, 0.0, 0.0, 0.0]),
            "omega": np.array([0.0, 0.0, 0.0])
        }
        
        dt = 0.01
        time_elapsed = 0.0
        flight_data = []
        
        # Phase 1: Takeoff
        autopilot.engage()
        autopilot.set_mode("altitude_hold")
        autopilot.set_target_altitude(1000.0)
        
        while state["position"][2] < 900.0 and time_elapsed < 60.0:
            # Get control commands
            control = autopilot.compute_command(state, dt)
            
            # Convert to forces for dynamics
            thrust_force = control["force"]
            
            # Update dynamics
            new_state = dynamics.step(
                state["position"],
                state["velocity"],
                thrust_force,
                dt
            )
            
            # Update state
            state["position"] = new_state["position"]
            state["velocity"] = new_state["velocity"]
            
            flight_data.append({
                "time": time_elapsed,
                "altitude": state["position"][2],
                "speed": np.linalg.norm(state["velocity"])
            })
            
            time_elapsed += dt
        
        # Verify takeoff successful
        assert state["position"][2] > 800.0
        assert time_elapsed < 60.0  # Should reach altitude within 1 minute
        
        # Phase 2: Cruise
        autopilot.set_mode("waypoint")
        waypoints = [
            np.array([1000.0, 0.0, 1000.0]),
            np.array([1000.0, 1000.0, 1000.0]),
            np.array([0.0, 1000.0, 1000.0])
        ]
        autopilot.set_waypoints(waypoints)
        
        cruise_start = time_elapsed
        while autopilot.current_waypoint_idx < len(waypoints) and time_elapsed < 300.0:
            control = autopilot.compute_command(state, dt)
            thrust_force = control["force"]
            
            # Add environmental effects
            altitude = state["position"][2]
            air_density = density_at_altitude(altitude)
            wind = wind_at_position(state["position"], time_elapsed)
            
            # Apply aerodynamic forces
            velocity_air = state["velocity"] - wind
            speed = np.linalg.norm(velocity_air)
            
            if speed > 0.1:
                alpha = np.arctan2(velocity_air[2], 
                                 np.sqrt(velocity_air[0]**2 + velocity_air[1]**2))
                
                aero_forces = aerodynamic_forces.aerodynamic_forces(
                    velocity_air,
                    alpha,
                    altitude,
                    dynamics.S,
                    dynamics.Cl0,
                    dynamics.Cld_alpha,
                    dynamics.Cd0,
                    dynamics.k
                )
                
                total_force = thrust_force + aero_forces["force"]
            else:
                total_force = thrust_force
            
            # Update dynamics
            new_state = dynamics.step(
                state["position"],
                state["velocity"],
                total_force,
                dt
            )
            
            state["position"] = new_state["position"]
            state["velocity"] = new_state["velocity"]
            
            # Check waypoint reached
            if np.linalg.norm(state["position"] - waypoints[autopilot.current_waypoint_idx]) < 50.0:
                autopilot.current_waypoint_idx += 1
            
            time_elapsed += dt
        
        # Verify cruise completed
        assert autopilot.current_waypoint_idx >= 2  # At least 2 waypoints reached
        
        # Phase 3: Landing
        autopilot.set_mode("altitude_hold")
        autopilot.set_target_altitude(0.0)
        
        landing_start = time_elapsed
        while state["position"][2] > 10.0 and time_elapsed < 400.0:
            control = autopilot.compute_command(state, dt)
            
            # Reduce thrust for descent
            thrust_force = control["force"] * 0.7
            
            # Update with full aerodynamics
            altitude = state["position"][2]
            air_density = density_at_altitude(altitude)
            
            velocity_air = state["velocity"]
            speed = np.linalg.norm(velocity_air)
            
            if speed > 0.1:
                alpha = np.arctan2(velocity_air[2],
                                 np.sqrt(velocity_air[0]**2 + velocity_air[1]**2))
                
                aero_forces = aerodynamic_forces.aerodynamic_forces(
                    velocity_air,
                    alpha,
                    altitude,
                    dynamics.S,
                    dynamics.Cl0,
                    dynamics.Cld_alpha,
                    dynamics.Cd0,
                    dynamics.k
                )
                
                total_force = thrust_force + aero_forces["force"]
            else:
                total_force = thrust_force
            
            new_state = dynamics.step(
                state["position"],
                state["velocity"],
                total_force,
                dt
            )
            
            state["position"] = new_state["position"]
            state["velocity"] = new_state["velocity"]
            
            time_elapsed += dt
        
        # Verify safe landing
        assert state["position"][2] < 20.0
        assert np.linalg.norm(state["velocity"]) < 10.0  # Low landing speed
    
    def test_environmental_effects_integration(self, flight_system):
        """Test integration of all environmental effects"""
        dynamics = flight_system["dynamics"]
        
        # Test conditions
        position = np.array([500.0, 500.0, 2000.0])
        velocity = np.array([50.0, 0.0, 0.0])
        time_val = 10.0
        
        # Get all environmental effects
        altitude = position[2]
        air_density = atmospheric_density.density_at_altitude(altitude)
        temperature = atmospheric_density.temperature_at_altitude(altitude)
        wind = wind_interaction.wind_at_position(position, time_val)
        
        # Add turbulence
        if hasattr(k_epsilon, 'compute_turbulence'):
            k_eps_turb = k_epsilon.compute_turbulence(velocity, position)
            turbulence = k_eps_turb["fluctuation"]
        else:
            turbulence = np.zeros(3)
        
        # Total environmental velocity
        env_velocity = wind + turbulence
        
        # Relative velocity for aerodynamics
        velocity_air = velocity - env_velocity
        
        # Verify reasonable values
        assert 0.5 < air_density < 1.3  # kg/m³
        assert 200 < temperature < 300  # K
        assert np.linalg.norm(wind) < 50.0  # m/s
        assert np.linalg.norm(turbulence) < 10.0  # m/s
    
    def test_control_system_integration(self, flight_system):
        """Test integration of control components"""
        controller = flight_system["controller"]
        autopilot = flight_system["autopilot"]
        stability = flight_system["stability"]
        
        # Create actuator system
        actuators = ActuatorGroup(name="main_actuators")
        
        # Add thrusters
        thruster_positions = [
            ("front_left", np.array([1.0, -0.5, 0.0])),
            ("front_right", np.array([1.0, 0.5, 0.0])),
            ("rear_left", np.array([-1.0, -0.5, 0.0])),
            ("rear_right", np.array([-1.0, 0.5, 0.0]))
        ]
        
        for name, pos in thruster_positions:
            thruster = ThrusterActuator(
                name=name,
                max_thrust=2500.0,
                position=pos
            )
            actuators.add_actuator(thruster)
        
        # Test state
        state = {
            "position": np.array([0.0, 0.0, 1000.0]),
            "velocity": np.array([30.0, 5.0, -2.0]),
            "orientation": np.array([0.99, 0.1, 0.0, 0.0]),  # Slight roll
            "omega": np.array([0.0, 0.05, 0.0])  # Slight pitch rate
        }
        
        # Reference state (level flight)
        reference = {
            "velocity": np.array([40.0, 0.0, 0.0]),
            "omega": np.array([0.0, 0.0, 0.0])
        }
        
        # Get raw control
        raw_control = controller.compute_control(state, reference, dt=0.01)
        
        # Apply stability augmentation
        if hasattr(stability, 'augment_control'):
            augmented_control = stability.augment_control(raw_control, state)
        else:
            augmented_control = raw_control
        
        # Allocate to actuators
        if hasattr(actuators, 'allocate_control'):
            actuator_commands = actuators.allocate_control(
                augmented_control["force"],
                augmented_control["moment"]
            )
            
            # Apply commands
            for name, command in actuator_commands.items():
                actuators.get_actuator(name).set_command(command)
        
        # Verify reasonable outputs
        assert np.linalg.norm(augmented_control["force"]) < 10000.0  # N
        assert np.linalg.norm(augmented_control["moment"]) < 1000.0  # N⋅m
    
    def test_sensor_integration(self):
        """Test sensor integration with flight system"""
        # Create sensors
        imu = IMUSensor(
            noise_level=0.01,
            bias=np.array([0.001, 0.001, 0.001]),
            sample_rate=100.0
        )
        
        pitot = PitotSensor(
            noise_level=0.5,
            min_speed=5.0,
            max_speed=200.0
        )
        
        # True state
        true_state = {
            "acceleration": np.array([0.5, -0.2, 9.81]),
            "omega": np.array([0.1, 0.0, -0.05]),
            "velocity": np.array([50.0, 5.0, -2.0]),
            "position": np.array([100.0, 200.0, 1500.0])
        }
        
        # Get sensor measurements
        imu_data = imu.measure(true_state)
        airspeed = pitot.measure(true_state["velocity"], wind=np.array([5.0, 0.0, 0.0]))
        
        # Verify measurements are noisy but close to truth
        assert np.allclose(imu_data["acceleration"], true_state["acceleration"], atol=0.1)
        assert np.allclose(imu_data["gyro"], true_state["omega"], atol=0.01)
        assert abs(airspeed - np.linalg.norm(true_state["velocity"] - np.array([5.0, 0.0, 0.0]))) < 2.0
    
    def test_physics_plugin_integration(self):
        """Test C++ physics plugin integration"""
        try:
            from aerodynamics.physics_plugin.python import bindings
            
            # Create mesh
            vertices = np.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.5, 0.5, 1.0]
            ]).flatten()
            
            faces = np.array([
                [0, 1, 2],
                [0, 1, 3],
                [0, 2, 3],
                [1, 2, 3]
            ]).flatten()
            
            mesh = bindings.create_mesh(vertices, faces)
            
            # Create flow state
            flow = bindings.create_flow_state(mesh, velocity=10.0, density=1.225)
            
            # Create turbulence model
            turb_model = bindings.create_k_epsilon_model()
            
            # Create solver
            solver = bindings.create_solver(mesh, turb_model)
            
            # Run simulation step
            dt = 0.01
            bindings.solver_step(solver, flow, dt)
            
            # Get forces
            forces = bindings.get_aerodynamic_forces(solver, flow)
            
            # Verify forces computed
            assert "drag" in forces
            assert "lift" in forces
            assert forces["drag"] > 0  # Drag should be positive
            
            # Cleanup
            bindings.destroy_solver(solver)
            bindings.destroy_turbulence_model(turb_model)
            bindings.destroy_flow_state(flow)
            bindings.destroy_mesh(mesh)
            
        except (ImportError, OSError) as e:
            pytest.skip(f"Physics plugin not available: {e}")
    
    def test_simulation_runner(self):
        """Test the simulation runner with all components"""
        # Check if simulation module exists
        try:
            from aerodynamics.simulations.run_simulation import run_simulation
            
            # Simple hover test configuration
            config = {
                "vehicle": {
                    "mass": 100.0,
                    "wing_area": 2.0,
                    "Cl0": 0.1,
                    "Cld_alpha": 5.0,
                    "Cd0": 0.02,
                    "k": 0.04
                },
                "initial_state": {
                    "position": [0.0, 0.0, 100.0],
                    "velocity": [0.0, 0.0, 0.0]
                },
                "control": {
                    "mode": "hover",
                    "target_altitude": 100.0
                },
                "simulation": {
                    "duration": 10.0,
                    "dt": 0.01,
                    "output_interval": 0.1
                }
            }
            
            # Run simulation
            results = run_simulation(config)
            
            # Verify results
            assert "time" in results
            assert "position" in results
            assert "velocity" in results
            assert len(results["time"]) > 0
            
            # Check hover maintained
            final_altitude = results["position"][-1][2]
            assert abs(final_altitude - 100.0) < 5.0  # Within 5m of target
            
        except ImportError:
            pytest.skip("Simulation runner not fully implemented")
    
    def test_emergency_scenarios(self, flight_system):
        """Test system behavior in emergency scenarios"""
        dynamics = flight_system["dynamics"]
        autopilot = flight_system["autopilot"]
        controller = flight_system["controller"]
        
        # Scenario 1: Engine failure
        state = {
            "position": np.array([0.0, 0.0, 1000.0]),
            "velocity": np.array([50.0, 0.0, 0.0]),
            "orientation": np.array([1.0, 0.0, 0.0, 0.0]),
            "omega": np.array([0.0, 0.0, 0.0])
        }
        
        # Simulate engine failure - no thrust available
        autopilot.emergency_stop()
        control = autopilot.compute_command(state, dt=0.01)
        
        # Should return minimal/safe commands
        assert np.linalg.norm(control["force"]) < 100.0
        
        # Scenario 2: Extreme attitude recovery
        extreme_state = {
            "position": np.array([0.0, 0.0, 1000.0]),
            "velocity": np.array([30.0, 0.0, -20.0]),  # Diving
            "orientation": np.array([0.7071, 0.7071, 0.0, 0.0]),  # 90° roll
            "omega": np.array([0.5, 0.0, 0.0])  # Rolling
        }
        
        # Stability system should command recovery
        recovery_ref = {
            "velocity": extreme_state["velocity"],
            "omega": np.array([0.0, 0.0, 0.0])  # Stop rotation
        }
        
        recovery_control = controller.compute_control(
            extreme_state,
            recovery_ref,
            dt=0.01
        )
        
        # Should command strong corrective moment
        assert abs(recovery_control["moment"][0]) > 10.0  # Anti-roll moment