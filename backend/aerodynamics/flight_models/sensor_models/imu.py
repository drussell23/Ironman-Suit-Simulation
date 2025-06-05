import numpy as np
import logging

logger = logging.getLogger(__name__)

class IMUSensor:
    """
    IMU sensor model: simulates 3-axis accelerometer and gyroscope measurements
    with configurable noise, bias, scale factor, and misalignment errors.
    """

    def __init__(
        self,
        noise_std_accel: float = 0.02,
        noise_std_gyro: float = 0.001,
        bias_accel: np.ndarray = None,
        bias_gyro: np.ndarray = None,
        scale_accel: np.ndarray = None,
        scale_gyro: np.ndarray = None,
        misalignment: np.ndarray = None,
    ):
        self.noise_std_accel = noise_std_accel
        self.noise_std_gyro = noise_std_gyro
        # Initialize biases (3-vector)
        self.bias_accel = np.zeros(3) if bias_accel is None else np.asarray(bias_accel, float)
        self.bias_gyro = np.zeros(3) if bias_gyro is None else np.asarray(bias_gyro, float)
        # Scale factors per axis
        self.scale_accel = np.ones(3) if scale_accel is None else np.asarray(scale_accel, float)
        self.scale_gyro = np.ones(3) if scale_gyro is None else np.asarray(scale_gyro, float)
        # Misalignment matrix (3x3)
        self.misalignment = np.eye(3) if misalignment is None else np.asarray(misalignment, float)

    def measure_acceleration(self, true_accel: np.ndarray) -> np.ndarray:
        """
        Measure body-frame acceleration (m/s^2) including sensor errors.

        :param true_accel: true acceleration vector [ax, ay, az]
        :return: measured acceleration (3-vector)
        """
        a = np.asarray(true_accel, float)
        if a.shape != (3,):
            raise ValueError("true_accel must be a 3-element vector")
        # Apply scale factors and misalignment
        a_scaled = self.scale_accel * a
        a_err = self.misalignment.dot(a_scaled)
        # Add bias and noise
        noise = np.random.normal(0.0, self.noise_std_accel, size=3)
        meas = a_err + self.bias_accel + noise
        logger.debug(f"Accel meas: true={a}, meas={meas}")
        return meas

    def measure_angular_rate(self, true_gyro: np.ndarray) -> np.ndarray:
        """
        Measure body-frame angular rate (rad/s) including sensor errors.

        :param true_gyro: true angular rate vector [p, q, r]
        :return: measured angular rate (3-vector)
        """
        g = np.asarray(true_gyro, float)
        if g.shape != (3,):
            raise ValueError("true_gyro must be a 3-element vector")
        # Apply scale factors and misalignment
        g_scaled = self.scale_gyro * g
        g_err = self.misalignment.dot(g_scaled)
        # Add bias and noise
        noise = np.random.normal(0.0, self.noise_std_gyro, size=3)
        meas = g_err + self.bias_gyro + noise
        logger.debug(f"Gyro meas: true={g}, meas={meas}")
        return meas