#!/usr/bin/env python3
"""
run_control.py

Integration script to run the Autopilot loop with dummy sensor data.
"""
import time
import numpy as np
import logging

from backend.aerodynamics.control.autopilot import Autopilot, SensorError, GuidanceError, ControllerError, ActuatorError
from backend.aerodynamics.control.guidance import Guidance
from backend.aerodynamics.control.controller import Controller
from backend.aerodynamics.control.actuator import Actuator, Thruster

logger = logging.getLogger(__name__)

class DummySensor:
    """Simulates a sensor providing position, velocity, and angular velocity."""
    def __init__(self):
        self._pos = np.zeros(3)
        self._vel = np.zeros(3)
        self._omega = np.zeros(3)

    def read_state(self):
        # In a real setting, update self._pos, self._vel, self._omega
        return {
            'position': self._pos.tolist(),
            'velocity': self._vel.tolist(),
            'angular_velocity': self._omega.tolist(),
            'acceleration': np.zeros(3).tolist(),
            'angular_acceleration': np.zeros(3).tolist()
        }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s][%(name)s][%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )

    # Define waypoints for guidance
    waypoints = [[0, 0, 0], [5, 0, 0], [5, 5, 0], [0, 5, 0]]

    sensor = DummySensor()
    guidance = Guidance(waypoints, speed=1.0, acceptance_radius=0.5)
    controller = Controller(mass=1.0, kp_trans=1.0, kd_trans=0.1, kp_rot=1.0, kd_rot=0.1)

    # Setup a basic 4-thruster actuator
    thrusters = [
        Thruster([1, 0, 0], [1, 0, 0], max_thrust=10),
        Thruster([-1, 0, 0], [-1, 0, 0], max_thrust=10),
        Thruster([0, 1, 0], [0, 1, 0], max_thrust=10),
        Thruster([0, -1, 0], [0, -1, 0], max_thrust=10),
    ]
    actuator = Actuator(thrusters)

    ap = Autopilot(sensor, guidance, controller, actuator, enable_logging=True, enable_history=True)

    try:
        ap.run(duration=5.0, rate_hz=10.0, log_every_n=1)
    except (SensorError, GuidanceError, ControllerError, ActuatorError) as e:
        logger.error(f"Autopilot run failed: {e}")

    # Print history
    for entry in ap.get_history():
        print(entry)

if __name__ == '__main__':
    main()