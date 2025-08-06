"""
Unit and physical conversion utilities for aerodynamics.
"""
import math
import numpy as np
from .constants import g0, R, gamma, P0, T0, rho0, L, a0


def deg2rad(deg: float) -> float:
    """
    Convert degrees to radians.

    Raises:
        ValueError for invalid input.
    """
    try:
        return math.radians(deg)
    except Exception as e:
        raise ValueError(f"deg2rad: invalid input {deg}: {e}")


def rad2deg(rad: float) -> float:
    """
    Convert radians to degrees.
    """
    try:
        return math.degrees(rad)
    except Exception as e:
        raise ValueError(f"rad2deg: invalid input {rad}: {e}")


def mps_to_kts(mps: float) -> float:
    """
    Convert meters per second to knots.
    """
    return mps * 1.9438445


def kts_to_mps(kts: float) -> float:
    """
    Convert knots to meters per second.
    """
    return kts * 0.514444


def mps_to_kmh(mps: float) -> float:
    """
    Convert meters per second to kilometers per hour.
    """
    return mps * 3.6


def kmh_to_mps(kmh: float) -> float:
    """
    Convert kilometers per hour to meters per second.
    """
    return kmh / 3.6


def altitude_to_pressure(altitude: float) -> float:
    """
    Compute barometric pressure at a given altitude using the international standard atmosphere.

    Args:
        altitude: geometric altitude above sea level in meters.
    Returns:
        Pressure in Pascals.
    """
    T = T0 - L * altitude
    if T <= 0:
        raise ValueError(f"altitude_to_pressure: resulting temperature {T} K <= 0")
    return P0 * (T / T0) ** (g0 / (R * L))


def pressure_to_altitude(pressure: float) -> float:
    """
    Invert barometric formula to compute altitude from pressure.

    Args:
        pressure: pressure in Pascals.
    Returns:
        Altitude in meters.
    """
    if pressure <= 0 or pressure > P0:
        raise ValueError(f"pressure_to_altitude: invalid pressure {pressure}")
    return (T0 / L) * (1 - (pressure / P0) ** (R * L / g0))


def density_at_altitude(altitude: float) -> float:
    """
    Compute air density at a given altitude using standard atmosphere.

    Returns:
        Density in kg/m^3.
    """
    P = altitude_to_pressure(altitude)
    T = T0 - L * altitude
    return P / (R * T)


def speed_of_sound(temperature: float = T0) -> float:
    """
    Compute speed of sound at a given temperature.

    Returns:
        Speed of sound in m/s.
    """
    return math.sqrt(gamma * R * temperature)