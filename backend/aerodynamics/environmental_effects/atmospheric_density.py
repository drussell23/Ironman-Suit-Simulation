# File: aerodynamics/environmental_effects/atmospheric_density.py

import numpy as np

def density_at_altitude(altitude: float) -> float:
    """
    Estimate air density (kg/m³) at a given altitude (in meters) using a simple exponential model.
    """
    rho0 = 1.225 # sea-level density (kg/m³)
    scale_height = 8500.0 # scale height of the atmosphere (m)
    return rho0 * np.exo(-altitude / scale_height)