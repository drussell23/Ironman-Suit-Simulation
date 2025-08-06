# File: backend/aerodynamics/flight_models/stability_control.py

"""
_summary_
Simple stability control module for the Iron Main suit:
- Implements a PD controller for angular stabilization (roll, pitch, yaw)
- Computes control torques based on desired vs. actual orientation and angular rates.
"""
import numpy as np


class StabilityControl:
    def __init__(
        self,
        Kp: np.ndarray = np.array(
            [10.0, 10.0, 10.0]
        ),  # Proportional gains [roll, pitch, yaw]
        Kd: np.ndarray = np.array(
            [5.0, 5.0, 5.0]
        ),  # Derivative gains [roll, pitch, yaw]
    ):
        """
        :param Kp: Proportional gain vector [Kp_roll, Kp_pitch, Kp_yaw]
        :param Kd: Derviative gain vector [Kd_roll, Kd_pitch, Kd_yaw]
        """
        self.Kp = Kp
        self.Kd = Kd

    def compute_torques(
        self, orientation_error: np.ndarray, angular_rate_error: np.ndarray
    ) -> np.ndarray:
        """_summary_
        Compute control torques to stabilize the suit.

        Args:
            orientation_error (np.ndarray): Error in Euler angles [roll_err, pitch_err, yaw_err]
            angular_rate_error (np.ndarray): Error in angular rates [p_err, q_err, r_err] (rad/s)

        Returns:
            np.ndarray: Torque vector [tau_roll, tau_pitch, tau_yaw]
        """
        # PD control: tau = Kp * error + Kd * error_rate
        torque = self.Kp * orientation_error + self.Kd * angular_rate_error
        return torque

    def stabilize(
        self,
        current_orientation: np.ndarray,
        current_rates: np.ndarray,
        desired_orientation: np.ndarray = np.zeros(3),
        desired_rates: np.ndarray = np.zeros(3),
    ) -> np.ndarray:
        """_summary_
        High-level helper to compute orientation and rate errors, then torques.

        Args:
            current_orientation (np.ndarray): Euler angles [roll, pitch, yaw] (rad)
            current_rates (np.ndarray): Angular rates [p, q, r] (rad/s)
            desired_orientation (np.ndarray, optional): Desired Euler angles (default hover = [0, 0, 0]). Defaults to np.zeros(3).
            desired_rates (np.ndarray, optional): Desired angular rates (default stationary = [0, 0, 0]). Defaults to np.zeros(3).

        Returns:
            np.ndarray: Torque vector [tau_roll, tau_pitch, tau_yaw]
        """
        # Compute orientation and rate errors.
        orientation_error = desired_orientation - current_orientation
        rate_error = desired_rates - current_rates

        # Wrap yaw error into [-pi, pi].
        orientation_error[2] = (orientation_error[2] + np.pi) % (2 * np.pi) - np.pi

        return self.compute_torques(orientation_error, rate_error)
