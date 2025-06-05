import numpy as np
import pytest

from backend.aerodynamics.flight_models.sensor_models.imu import IMUSensor

"""
IMU sensor model tests 

Test cases:
    1. Accelerometer without errors returns true
    2. Accelerometer with scale and bias
    3. Accelerometer with misalignment
    4. Accelerometer invalid shape raises
    5. Gyroscope without errors returns true
    6. Gyroscope with scale and bias
    7. Gyroscope invalid shape raises 
"""

# Test case 1: Accelerometer without errors returns true
def test_accel_without_errors_returns_true():
    imu = IMUSensor(
        noise_std_accel=0.0,
        bias_accel=np.zeros(3),
        scale_accel=np.ones(3),
        misalignment=np.eye(3),
    )
    true_accel = np.array([0.0, 9.81, 0.0])
    meas = imu.measure_acceleration(true_accel)
    assert np.allclose(meas, true_accel)

# Test case 2: Accelerometer with scale and bias
def test_accel_with_scale_and_bias():
    scale = np.array([1.1, 0.9, 1.05])
    bias = np.array([0.1, -0.1, 0.05])
    imu = IMUSensor(
        noise_std_accel=0.0,
        bias_accel=bias,
        scale_accel=scale,
        misalignment=np.eye(3),
    )
    true_accel = np.array([1.0, 2.0, 3.0])
    expected = scale * true_accel + bias
    meas = imu.measure_acceleration(true_accel)
    assert np.allclose(meas, expected)

# Test case 3: Accelerometer with misalignment
def test_accel_with_misalignment():
    # 90° rotation about Z: x->y, y->-x
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    imu = IMUSensor(
        noise_std_accel=0.0,
        bias_accel=np.zeros(3),
        scale_accel=np.ones(3),
        misalignment=R,
    )
    true_accel = np.array([1.0, 0.0, 0.0])
    meas = imu.measure_acceleration(true_accel)
    assert np.allclose(meas, np.array([0.0, 1.0, 0.0]))

# Test case 4: Accelerometer invalid shape raises
def test_accel_invalid_shape_raises():
    imu = IMUSensor()
    with pytest.raises(ValueError):
        imu.measure_acceleration(np.array([1.0, 2.0]))

# Test case 5: Gyroscope without errors returns true
def test_gyro_without_errors_returns_true():
    imu = IMUSensor(
        noise_std_gyro=0.0,
        bias_gyro=np.zeros(3),
        scale_gyro=np.ones(3),
        misalignment=np.eye(3),
    )
    true_gyro = np.array([0.1, -0.2, 0.3])
    meas = imu.measure_angular_rate(true_gyro)
    assert np.allclose(meas, true_gyro)

# Test case 6: Gyroscope with scale and bias
def test_gyro_with_scale_and_bias():
    scale = np.array([0.98, 1.02, 1.05])
    bias = np.array([-0.01, 0.02, -0.03])
    imu = IMUSensor(
        noise_std_gyro=0.0,
        bias_gyro=bias,
        scale_gyro=scale,
        misalignment=np.eye(3),
    )
    true_gyro = np.array([0.5, -0.5, 0.5])
    expected = scale * true_gyro + bias
    meas = imu.measure_angular_rate(true_gyro)
    assert np.allclose(meas, expected)

# Test case 7: Gyroscope invalid shape raises
def test_gyro_invalid_shape_raises():
    imu = IMUSensor()
    with pytest.raises(ValueError):
        imu.measure_angular_rate(np.array([0.1, 0.2]))