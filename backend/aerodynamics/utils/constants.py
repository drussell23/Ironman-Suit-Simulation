"""
Physical constants for aerodynamics and atmospheric calculations.
"""
import math

# Standard gravitational acceleration (m/s^2)
g0: float = 9.80665

# Specific gas constant for dry air (J/(kg·K))
R: float = 287.05

# Ratio of specific heats for air (unitless)
gamma: float = 1.4

# Standard sea-level conditions
P0: float = 101325.0    # Pa
T0: float = 288.15      # K
rho0: float = 1.225     # kg/m^3

# Standard atmosphere lapse rate (K/m)
L: float = 0.0065

# Speed of sound at sea level (m/s)
a0: float = math.sqrt(gamma * R * T0)