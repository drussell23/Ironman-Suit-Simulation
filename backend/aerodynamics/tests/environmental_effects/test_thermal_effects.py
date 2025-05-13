import pytest
import math
from aerodynamics.environmental_effects.thermal_effects import (
    temperature_at_altitude,
    convective_heat_flux,
    radiative_heat_flux,
    net_heat_flux,
)
from aerodynamics.environmental_effects.atmospheric_density import (
    density_at_altitude,
)  # reference density lookup


# --- temperature_at_altitude tests ---
def test_temperature_at_altitude_profile_and_clamping():
    # Sea level
    assert temperature_at_altitude(0) == pytest.approx(288.15)
    # 1 km
    assert temperature_at_altitude(1_000) == pytest.approx(288.15 - 0.0065 * 1_000)
    # Tropopause
    tropo_temp = 288.15 - 0.0065 * 11_000
    assert temperature_at_altitude(11_000) == pytest.approx(tropo_temp)
    # Above tropopause stays constant
    assert temperature_at_altitude(20_000) == pytest.approx(tropo_temp)
    # Negative altitude clamps to sea level
    assert temperature_at_altitude(-100) == pytest.approx(288.15)


@pytest.mark.parametrize("bad_alt", ["a", None, (1,), []])
def test_temperature_at_altitude_invalid_type(bad_alt):
    with pytest.raises(TypeError):
        temperature_at_altitude(bad_alt)


# --- convective_heat_flux tests ---
def test_convective_heat_flux_known_value():
    # For velocity=100 m/s, altitude=0, radius=1m
    rho0 = density_at_altitude(0.0)
    expected = 1.83e-8 * math.sqrt(rho0 / 1.0) * (100.0**3)
    assert convective_heat_flux(100.0, 0.0, nose_radius_m=1.0) == pytest.approx(
        expected
    )


@pytest.mark.parametrize(
    "args",
    [
        ("100", 0.0, 0.1),
        (100.0, "0", 0.1),
        (100.0, 0.0, "0.1"),
    ],
)
def test_convective_heat_flux_type_errors(args):
    with pytest.raises(TypeError):
        convective_heat_flux(*args)


@pytest.mark.parametrize("r_n", [0, -0.5])
def test_convective_heat_flux_value_error(r_n):
    with pytest.raises(ValueError):
        convective_heat_flux(100.0, 0.0, nose_radius_m=r_n)


# --- radiative_heat_flux tests ---
def test_radiative_heat_flux_known_value_and_defaults():
    # T=300K, eps=1.0
    expected_full = 1.0 * 5.670374419e-8 * (300.0**4)
    assert radiative_heat_flux(300.0, emissivity=1.0) == pytest.approx(expected_full)
    # default emissivity = 0.8
    expected_def = 0.8 * 5.670374419e-8 * (300.0**4)
    assert radiative_heat_flux(300.0) == pytest.approx(expected_def)


@pytest.mark.parametrize("bad", [("300", 0.5), (300.0, "0.5")])
def test_radiative_heat_flux_type_errors(bad):
    with pytest.raises(TypeError):
        radiative_heat_flux(bad[0], emissivity=bad[1])


@pytest.mark.parametrize("eps", [-0.1, 1.1])
def test_radiative_heat_flux_value_errors(eps):
    with pytest.raises(ValueError):
        radiative_heat_flux(300.0, emissivity=eps)


# --- net_heat_flux tests ---
def test_net_heat_flux_combines_conv_and_rad():
    q_conv = convective_heat_flux(100.0, 0.0, nose_radius_m=1.0)
    q_rad = radiative_heat_flux(300.0, emissivity=1.0)
    assert net_heat_flux(
        100.0, 0.0, 300.0, nose_radius_m=1.0, emissivity=1.0
    ) == pytest.approx(q_conv - q_rad)
