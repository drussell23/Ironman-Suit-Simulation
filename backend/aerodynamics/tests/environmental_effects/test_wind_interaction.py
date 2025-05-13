import pytest
import numpy as np
import math
import random
import time
from aerodynamics.environmental_effects.wind_interaction import wind_at_position  # citeturn3file1

@pytest.fixture(autouse=True)
def disable_turbulence(monkeypatch):
    """
    By default, monkeypatch random.gauss to return zero turbulence for deterministic tests.
    """
    monkeypatch.setattr(random, "gauss", lambda mu, sigma: 0.0)


def test_wind_at_ground_zero_shear_and_gust():
    """
    Ground-level wind at (0,0,0) and t=0 combines base and gust correctly:
    - base_wx = 0, base_wz = 3
    - no shear at y=0
    - gust_x = 0, gust_z = 2
    => [0, 0, 5]
    """
    result = wind_at_position(0.0, 0.0, 0.0, t=0.0)
    expected = np.array([0.0, 0.0, 5.0])
    assert result.shape == (3,)
    assert pytest.approx(expected, rel=1e-6) == result


def test_negative_altitude_behavior():
    """
    Negative altitude (y < 0) should apply inverse shear factor:
    For y=-100 m:
    - base_wz = 3.0
    - wind_shear_factor = min(1.0, -0.1) = -0.1
    - wz_shear = 3.0 * (1 + 1.5 * (-0.1)) = 2.55
    - gust_z at t=0 = 2.0
    => total wz ~4.55
    """
    wind_neg = wind_at_position(0.0, -100.0, 0.0, t=0.0)
    assert wind_neg.shape == (3,)
    # Expect x=0, y=0, z≈4.55
    assert pytest.approx(0.0, rel=1e-6) == wind_neg[0]
    assert pytest.approx(0.0, rel=1e-6) == wind_neg[1]
    assert pytest.approx(4.55, rel=1e-6) == wind_neg[2]

@pytest.mark.parametrize("y1,y2", [(0.0, 500.0), (500.0, 1000.0), (1000.0, 2000.0)])
def test_shear_increases_with_altitude_until_saturation(y1, y2):
    """
    Vertical wind component should increase with altitude up to the shear cap.
    """
    wx1, wy1, wz1 = wind_at_position(0.0, y1, 0.0, t=0.0)
    wx2, wy2, wz2 = wind_at_position(0.0, y2, 0.0, t=0.0)
    assert wz2 >= wz1, f"Expected wz at y={y2} ({wz2}) >= wz at y={y1} ({wz1})"

def test_default_time_uses_current_time(monkeypatch):
    """
    If t is None, wind_at_position should use time.time().
    """
    fixed_t = 123.456
    monkeypatch.setattr(time, "time", lambda: fixed_t)
    expected = wind_at_position(10.0, 200.0, 5.0, t=fixed_t)
    result = wind_at_position(10.0, 200.0, 5.0)
    assert pytest.approx(expected, rel=1e-6) == result

@pytest.mark.parametrize("time_sec", [0.0, 15.0, 30.0, 45.0, 60.0])
def test_periodic_gusts_various_times(time_sec):
    """
    Gust components should follow a sine/cosine pattern over one period.
    """
    x, y, z = 0.0, 0.0, 0.0
    result = wind_at_position(x, y, z, t=time_sec)
    base_wx = 5.0 * math.sin(0.0005 * x) * math.cos(0.0005 * z)
    base_wz = 3.0 * math.cos(0.0005 * z)
    period = 60.0
    mg = 2.0
    phase = 2 * math.pi * (time_sec % period) / period
    gust_x = mg * math.sin(phase)
    gust_z = mg * math.cos(phase)
    expected = np.array([base_wx + gust_x, 0.0, base_wz + gust_z])
    assert pytest.approx(expected, rel=1e-6) == result

def test_spatial_variation_effect():
    """
    Wind should vary spatially with x and z according to the base model.
    """
    x, y, z, t = 1000.0, 0.0, 500.0, 0.0
    result = wind_at_position(x, y, z, t=t)
    base_wx = 5.0 * math.sin(0.0005 * x) * math.cos(0.0005 * z)
    base_wy = 0.0
    base_wz = 3.0 * math.cos(0.0005 * z)
    gust_z = 2.0
    expected = np.array([base_wx, base_wy, base_wz + gust_z])
    assert pytest.approx(expected, rel=1e-6) == result

@pytest.mark.parametrize("bad_input", ["str", [], {}, (1,2)])
def test_invalid_input_raises_type_error(bad_input):
    """
    Non-numeric or non-scalar inputs should raise a TypeError.
    """
    with pytest.raises(TypeError):
        wind_at_position(bad_input, 0.0, 0.0, t=0.0)
    with pytest.raises(TypeError):
        wind_at_position(0.0, bad_input, 0.0, t=0.0)
    with pytest.raises(TypeError):
        wind_at_position(0.0, 0.0, bad_input, t=0.0)
    with pytest.raises(TypeError):
        wind_at_position(0.0, 0.0, 0.0, t=bad_input)

def test_integer_inputs_cast_to_float():
    """
    Integer inputs should be accepted (treated as floats).
    """
    int_res = wind_at_position(5, 10, 15, t=20)
    float_res = wind_at_position(5.0, 10.0, 15.0, t=20.0)
    assert np.allclose(int_res, float_res)
