# File: backend/aerodynamics/environmental_effects/atmospheric_density.py

"""
Simple atmospheric density model using an exponential atmosphere assumption.
"""
import numpy as np 

# Sea-level standard atmospheric density (kg/m^3)
SEA_LEVEL_DENSITY = 1.225
# Scale height for Earth's atmosphere (m)
SCALE_HEIGHT = 8500.0

def density_at_altitude(altitude: float) -> float:
    """
    Calculate the air density at a given altitude using an exponential model. 
    
    Args:
        altitude (float): Altitude above sea level in meters.

    Returns:
        float: Air density (kg/m^3)
    """
    # Ensure altitude is non-negative.
    h = max(0.0, altitude)
    
    # Exponential decrease of density with altitude.
    return SEA_LEVEL_DENSITY * np.exp(-h / SCALE_HEIGHT)