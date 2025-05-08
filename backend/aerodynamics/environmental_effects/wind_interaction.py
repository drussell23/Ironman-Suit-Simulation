# File: aerodynamics/environmental_effects/wind_interaction.py
import numpy as np

def wind_at_position(x: float, z: float) -> np.ndarray:
    """
    Simple sinusoidal wind field varying with x and z.
    Returns a 3D wind velocity vector (m/s).
    """
    # Example: wind blowing in +X, with strength varying on a sine wave in x & z
    speed = 5.0 * np.sin(0.001 * x) * np.cos(0.001 * z)
    return np.array([speed, 0.0, 0.0])