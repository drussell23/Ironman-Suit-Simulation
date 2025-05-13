"""_summary_
Thermal effects module for Iron Man suit simulation.
Provides standard atmosphere tempature profile and heat flux estimations.
"""

import math
from .atmospheric_density import density_at_altitude

# Standard atmosphere constants.
T0 = 288.15  # Sea level temperature in Kelvin (K).
LAPSE_RATE = 0.0065  # Temperature lapse rate in K/m.
TROPOPAUSE_ALTITUDE = 11_000.0  # Altitude of tropopause in meters (m).
T_TROPOPAUSE = T0 - LAPSE_RATE * TROPOPAUSE_ALTITUDE  # Temperature at tropopause in K.

# Heat flux model constants.
STAGNATION_HEATING_CONST = 1.83e-8  # Sutton-Graves constant, SI units (W/m^2/K).
STEFAN_BOLTZMANN = 5.670374419e-8  # Stefan-Boltzmann constant, W/(m^2 K^4)


def temperature_at_altitude(altitude_m: float) -> float:
    """
    Compute ambient air temperature (K) at a given altitude using the
    International Standard Atmosphere up to the tropopause, then constant.

    Args:
        altitude_m (float): Altitude in meters. Values below 0 are clamped to 0.

    Returns:
        float: Ambient air temperature in Kelvin.

    Raises:
        TypeError: If altitude_m is not a number.
    """
    if not isinstance(altitude_m, (int, float)):
        raise TypeError("altitude_m must be a numeric type.")

    h = max(0.0, float(altitude_m))  # Clamp negative altitudes to sea level.

    if h <= TROPOPAUSE_ALTITUDE:
        return T0 - LAPSE_RATE * h  # Linear decrease with altitude.
    else:
        return T_TROPOPAUSE


def convective_heat_flux(
    velocity_m_s: float, altitude_m: float, nose_radius_m: float = 0.1
) -> float:
    """
    Estimate convective stagnation-point heat flux (W/m^2) using a simplified
    Sutton–Graves relation: q = k * sqrt(rho / r_n) * v^3.

    Args:
        velocity_m_s (float): Flow velocity relative to suit surface, m/s.
        altitude_m (float): Altitude in meters for density lookup.
        nose_radius_m (float): Effective nose radius in meters.

    Returns:
        float: Convective heat flux in W/m^2.

    Raises:
        TypeError: If inputs are not numeric.
        ValueError: If nose_radius_m is not positive.
    """
    # Input validation
    for name, val in (
        ("velocity_m_s", velocity_m_s),
        ("altitude_m", altitude_m),
        ("nose_radius_m", nose_radius_m),
    ):
        if not isinstance(val, (int, float)):
            raise TypeError(f"{name} must be a numeric type.")

    r_n = float(nose_radius_m)

    if r_n <= 0:
        raise ValueError("nose_radius_m must be positive.")

    v = float(velocity_m_s)
    rho = density_at_altitude(altitude_m)  # kg/m^3

    # Sutton-Graves stagnation heating.
    return STAGNATION_HEATING_CONST * math.sqrt(rho / r_n) * v**3


def radiative_heat_flux(temperature_surface_k: float, emissivity: float = 0.8) -> float:
    """
    Estimate convective stagnation-point heat flux (W/m^2) using a simplified
    Sutton–Graves relation: q = k * sqrt(rho / r_n) * v^3.

    Args:
        velocity_m_s (float): Flow velocity relative to suit surface, m/s.
        altitude_m (float): Altitude in meters for density lookup.
        nose_radius_m (float): Effective nose radius in meters.

    Returns:
        float: Convective heat flux in W/m^2.

    Raises:
        TypeError: If inputs are not numeric.
        ValueError: If nose_radius_m is not positive.
    """
    # Input validation.
    if not isinstance(temperature_surface_k, (int, float)):
        raise TypeError("temperature_surface_k must be a numeric type.")
    if not isinstance(emissivity, (int, float)):
        raise TypeError("emissivity must be a numeric type.")

    eps = float(emissivity)

    if not (0.0 <= eps <= 1.0):
        raise ValueError("emissivity must be between 0 and 1.")

    T = float(temperature_surface_k)

    return eps * STEFAN_BOLTZMANN * T**4


def net_heat_flux(
    velocity_m_s: float,
    altitude_m: float,
    surface_temperature_k: float,
    nose_radius_m: float = 0.1,
    emissivity: float = 0.8,
) -> float:
    """
    Compute net heat flux (W/m^2) at the suit surface: convective heating minus
    radiative cooling.

    Args:
        velocity_m_s (float): Flow velocity relative to suit, m/s.
        altitude_m (float): Altitude in meters.
        surface_temperature_k (float): Surface temperature in Kelvin.
        nose_radius_m (float): Effective nose radius, m.
        emissivity (float): Surface emissivity.

    Returns:
        float: Net heat flux (positive is heating).
    """
    q_conv = convective_heat_flux(velocity_m_s, altitude_m, nose_radius_m)
    q_rad = radiative_heat_flux(surface_temperature_k, emissivity)
    
    return q_conv - q_rad
