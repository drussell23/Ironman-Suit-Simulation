import numpy as np
import pytest

from backend.aerodynamics.flight_models.sensor_models.pitot import PitotSensor, SEA_LEVEL_DENSITY, GAS_CONSTANT
from backend.aerodynamics.environmental_effects.atmospheric_density import density_at_altitude
from backend.aerodynamics.environmental_effects.thermal_effects import temperature_at_altitude


def test_static_pressure_at_sea_level():
    pitot = PitotSensor(noise_std_static=0.0, bias_static=0.0)
    p_static = pitot.measure_static_pressure(0.0)
    rho = density_at_altitude(0.0)
    T = temperature_at_altitude(0.0)
    expected = rho * GAS_CONSTANT * T
    assert p_static == pytest.approx(expected, rel=1e-6)


def test_total_and_dynamic_pressure():
    pitot = PitotSensor(noise_std_static=0.0, bias_static=0.0,
                        noise_std_total=0.0, bias_total=0.0)
    velocity = np.array([10.0, 0.0, 0.0])
    p_s = pitot.measure_static_pressure(0.0)
    p_t = pitot.measure_total_pressure(velocity, 0.0)
    q_dyn = pitot.measure_dynamic_pressure(velocity, 0.0)
    expected_q = 0.5 * density_at_altitude(0.0) * 10.0**2
    assert p_t == pytest.approx(p_s + expected_q, rel=1e-6)
    assert q_dyn == pytest.approx(expected_q, rel=1e-6)


def test_indicated_airspeed_computation():
    pitot = PitotSensor()
    q = 0.5 * density_at_altitude(0.0) * 20.0**2
    v_ias = pitot.indicated_airspeed(q)
    expected = np.sqrt(2 * q / SEA_LEVEL_DENSITY)
    assert v_ias == pytest.approx(expected, rel=1e-6)


def test_true_airspeed_without_compressibility():
    pitot = PitotSensor()
    q = 0.5 * density_at_altitude(1000.0) * 15.0**2
    v_tas = pitot.true_airspeed(q, altitude=1000.0, compressibility=False)
    expected = np.sqrt(2 * q / density_at_altitude(1000.0))
    assert v_tas == pytest.approx(expected, rel=1e-6)


def test_true_airspeed_compressibility_low_mach():
    pitot = PitotSensor()
    q = 0.5 * density_at_altitude(5000.0) * 5.0**2
    v_no = pitot.true_airspeed(q, 5000.0, compressibility=False)
    v_co = pitot.true_airspeed(q, 5000.0, compressibility=True)
    assert v_co == pytest.approx(v_no, rel=1e-6)


def test_indicated_airspeed_negative_q():
    pitot = PitotSensor()
    v_ias = pitot.indicated_airspeed(-100.0)
    assert v_ias == 0.0


def test_measure_total_pressure_invalid_shape_raises():
    pitot = PitotSensor()
    with pytest.raises(ValueError):
        pitot.measure_total_pressure(np.array([1.0, 2.0]), 0.0)


def test_measure_dynamic_pressure_invalid_shape_raises():
    pitot = PitotSensor()
    with pytest.raises(ValueError):
        pitot.measure_dynamic_pressure(np.array([1.0, 2.0]), 0.0)