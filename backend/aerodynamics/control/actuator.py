"""
actuator.py

Provides Thruster and Actuator allocation models for drone/suit control.
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Thruster:
    def __init__(self, position, direction, max_thrust, time_constant=0.1):
        """
        :param position: 3D vector of thruster location relative to CG
        :param direction: 3D unit vector of thrust direction
        :param max_thrust: maximum thrust (N)
        :param time_constant: first-order lag time constant (s)
        """
        self.position = np.array(position, float)
        self.direction = np.array(direction, float) / np.linalg.norm(direction)
        self.max_thrust = max_thrust
        self.time_constant = time_constant
        self.command = 0.0
        self.current_thrust = 0.0

    """
    Set throttle command [0,1] 
    """
    def set_command(self, cmd):
        """Set throttle command [0,1]"""
        self.command = float(np.clip(cmd, 0.0, 1.0))
        logger.debug(f"Thruster cmd set to {self.command:.3f}")

    def update(self, dt):
        """Update thrust with first-order lag: dT/dt = (target - T)/tau"""
        target = self.command * self.max_thrust
        self.current_thrust += (target - self.current_thrust) / self.time_constant * dt
        logger.debug(f"Thruster thrust: {self.current_thrust:.2f} N at pos {self.position}")

    def force(self):
        """Return force vector (3,)"""
        return self.current_thrust * self.direction

    def moment(self):
        """Return moment vector r × F"""
        return np.cross(self.position, self.force())

class Actuator:
    def __init__(self, thrusters: list[Thruster]):
        """:param thrusters: list of Thruster objects"""
        self.thrusters = thrusters
        # Build allocation matrix A (6×N)
        N = len(thrusters)
        A = np.zeros((6, N))
        for i, t in enumerate(thrusters):
            A[0:3, i] = t.direction
            A[3:6, i] = np.cross(t.position, t.direction)
        self.allocation = A
        # Pseudo-inverse for under/over-actuated cases
        self.pinv = np.linalg.pinv(A)

    def allocate(self, force_cmd, moment_cmd):
        """
        Compute and send commands based on desired force and moment.
        :param force_cmd: (3,) array
        :param moment_cmd: (3,) array
        """
        cmd6 = np.concatenate([force_cmd, moment_cmd])
        thrusts = self.pinv.dot(cmd6)
        for thr, T in zip(self.thrusters, thrusts):
            thr.set_command(T / thr.max_thrust)

    def update(self, dt):
        """Update all thrusters dynamics."""
        for t in self.thrusters:
            t.update(dt)

    def get_total_force_and_moment(self):
        """Sum forces and moments from all thrusters."""
        F = np.zeros(3)
        M = np.zeros(3)
        for t in self.thrusters:
            F += t.force()
            M += t.moment()
        return F, M