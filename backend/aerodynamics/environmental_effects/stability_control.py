# File: backend/aerodynamics/flight_models/stability_control.py

"""
_summary_
Simple stability control module for the Iron Man suit:
- Implements a PD controller for angular stabilization (roll, pitch, yaw)
- Computes control torques based on desired vs. actual orientation and angular rates
"""
import numpy as np

class StabilityControl:
    def __init__(
        self, 
        Kp: np.ndarray = np.array([10.0, 10.0, 10.0]),
        Kd: np.ndarray = np.array([5.0, 5.0, 5.0])):
        """_summary_

        Args:
            Kp (np.ndarray, optional): Proportional gain vector [Kp_roll, Kp_pitch, Kp_yaw]. Defaults to np.array([10.0, 10.0, 10.0]).
            Kd (np.ndarray, optional): Derivative gain vector [Kd_roll, Kd_pitch, Kd_yaw]. Defaults to np.array([5.0, 5.0, 5.0]).
        """
        self.Kp = Kp 
        self.Kd = Kd
        
    def compute_torques(self, orientation_error: np.ndarray, angular_rate_error: np.ndarray) -> np.ndarray:
        """_summary_
        Compute control torques using a PD controller. 

        Args:
            orientation_error (np.ndarray): Error in Euler angles [roll_err, pitch_err, yaw_err] (rad)
            angular_rate_error (np.ndarray): Error in angular rates [p_err, q_err, r_err] (rad/s)

        Returns:
            np.ndarray: Torque vector [tau_roll, tau_pitch, tau_yaw]
        """
        # PD control: tau = Kp * error + Kd * error_rate
        return self.Kp * orientation_error + self.Kd * angular_rate_error
    
    def stabilize(self, current_orientation: np.ndarray, current_rates: np.ndarray, desired_orientation: np.ndarray = np.zeros(3), desired_rates: np.ndarray = np.zeros(3)) -> np.ndarray:
        """_summary_
        Calculate torques to stabilize the suit to desired orientation and rates.

        Args:
            current_orientation (np.ndarray): Euler angles [roll, pitch, yaw] (rad)
            current_rates (np.ndarray): Angular rates [p, q, r] (rad/s)
            desired_orientation (np.ndarray, optional): Desired Euler angles, defaults to [0, 0, 0]. Defaults to np.zeros(3).
            desired_rates (np.ndarray, optional): Desired angular rates, defaults to [0, 0, 0]. Defaults to np.zeros(3).

        Returns:
            np.ndarray: Torque vector [tau_roll, tau_pitch, tau_yaw]
        """
        # Compute errors
        orientation_error = desired_orientation - current_orientation
        rate_error = desired_rates - current_rates
        
        # Wrap yaw error to [-pi, pi]
        orientation_error[2] = (orientation_error[2] + np.pi) % (2 * np.pi) - np.pi 
        
        # Return PD control torques
        return self.compute_torques(orientation_error, rate_error)