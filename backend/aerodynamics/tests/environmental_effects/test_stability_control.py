import pytest
import numpy as np
from environmental_effects.stability_control import StabilityControl

# Default controller gains for PD control.
DEFAULT_KP = np.array([10.0, 10.0, 10.0])
DEFAULT_KD = np.array([5.0, 5.0, 5.0])

def test_compute_torques_default_gains():
    """
    _summary_
    PD control: torque = Kp * orientation_error + Kd * rate_error with default gains.
    """
    sc = StabilityControl()
    orientation_error = np.array([1.0, 2.0, 3.0])
    rate_error = np.array([0.1, 0.2, 0.3])
    expected = DEFAULT_KP * orientation_error + DEFAULT_KD * rate_error
    torque = sc.compute_torques(orientation_error, rate_error)
    
    # Verify element-wise equality within a small tolerance.
    assert torque.tolist() == pytest.approx(expected.tolist(), rel=1e-6), f"Expected {expected}, got {torque}"
    
def test_compute_torques_custom_gains():
    """
    _summary_
    PD control with user-specified proportional and derivative gains.
    """
    custom_Kp = np.array([1.0, 2.0, 3.0])
    custom_Kd = np.array([0.5, 1.0, 1.5])
    sc = StabilityControl(Kp=custom_Kp, Kd=custom_Kd)
    orientation_error = np.array([-1.0, 0.5, 2.0])
    rate_error = np.array([0.0, 0.1, -0.2])
    expected = custom_Kp * orientation_error + custom_Kd * rate_error
    torque = sc.compute_torques(orientation_error, rate_error)
    
    assert torque.tolist() == pytest.approx(expected.tolist(), rel=1e-6), f"Expected {expected}, got {torque}"
    
def test_stabilize_zero_error():
    """
    _summary_
    When the current orientation and rates match desired (default zeros), the computed torques should be zero.
    """
    sc = StabilityControl()
    current_orientation = np.zeros(3)
    current_rates = np.zeros(3)
    torque = sc.stabilize(current_orientation, current_rates)
    
    assert torque.tolist() == pytest.approx([0.0, 0.0, 0.0], rel=1e-6), f"Expected zero torque, got {torque}"