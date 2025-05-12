# File: backend/aerodynamics/environmental_effects/wind_interaction.py

"""
Advanced position- and altitude-based wind field simulation with turbulence
and temporal variations for the Iron Man suit simulation.
"""
import numpy as np
import math
import random
import time


def wind_at_position(x: float, y: float, z: float, t: float = None) -> np.ndarray:
    """
    Compute advanced wind velocity vector based on horizontal (x,z),
    altitude (y), and optional time (t).

    :param x: Horizontal x-coordinate in meters
    :param y: Altitude (vertical y-coordinate) in meters
    :param z: Horizontal z-coordinate in meters
    :param t: Simulation time in seconds (optional), for temporal variability
    :return: 3-element numpy array [wx, wy, wz] in m/s
    """
    # Base wind model: Spatially varying sinusoidal wind.
    base_wx = 5.0 * np.sin(0.0005 * x) * np.cos(0.0005 * z)
    base_wz = 3.0 * np.cos(0.0005 * z)
    base_wy = 0.0 # Typically minimal vertical wind unless modeling updrafts/downdrafts explicitly.
    
    # Altitude-dependent wind sher.
    wind_shear_factor = min(1.0, y / 1000.0)  # Linearly scale wind strength up to 1000m altitude. 
    wx_shear = base_wx * (1 + 2.0 * wind_shear_factor) # Increase wind speed with altitude. 
    wz_shear = base_wz * (1 + 1.5 * wind_shear_factor) # Increase wind speed with altitude.
    
    # Turbulence modeling (simple stochastic turbulence).
    turbulence_intensity = 0.1 # 10% of wind magnitude.
    turbulence_scale = turbulence_intensity * (np.sqrt(wx_shear ** 2 + wz_shear ** 2) + 1e-6)
    
    wx_turbulence = random.gauss(0, turbulence_scale) # Random Gaussian noise for wx.
    wz_turbulence = random.gauss(0, turbulence_scale) # Random Gaussian noise for wz.
    wy_turbulence = random.gauss(0, turbulence_scale / 2) # Less turbulence in vertical direction.
    
    # Temporal variability (periodic gusts).
    if t is None:
        t = time.time() # If time not provided, use current real-time for randomness.
        
    gust_period = 60.0 # Gust repeats roughly every 60 seconds.
    gust_magnitude = 2.0 # Gust wind speed variation in m/s.
    gust_phase = 2 * math.pi * (t % gust_period) / gust_period
    gust_wx = gust_magnitude * math.sin(gust_phase)
    gust_wz = gust_magnitude * math.cos(gust_phase)
    
    # Combine base wind, altitude shear, turbulence, and gusts.
    wx_total = wx_shear + wx_turbulence + gust_wx 
    wy_total = base_wy + wy_turbulence # Vertical wind component (usually small).
    wz_total = wz_shear + wz_turbulence + gust_wz
    
    return np.array([wx_total, wy_total, wz_total]) # Wind vector in m/s.
    
    
    
    
