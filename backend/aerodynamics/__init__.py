"""
Aerodynamics package for Iron Man Suit Simulation.

This package provides comprehensive aerodynamic modeling including:
- Environmental effects (atmosphere, wind, turbulence)
- Flight dynamics and control
- High-performance physics computations via C++ plugin
- Simulation and validation utilities
"""

# Make subpackages available at package level.
from . import environmental_effects
from . import flight_models
from . import control
from . import utils
from . import validation

# Import key classes and functions for convenience
from .flight_models import FlightDynamics, StabilityControl, compute_thrust
from .environmental_effects import density_at_altitude, wind_at_position
from .control.controller import Controller
from .control.autopilot import Autopilot

__all__ = [
    # Subpackages
    "environmental_effects",
    "flight_models",
    "control",
    "utils",
    "validation",
    # Key classes
    "FlightDynamics",
    "StabilityControl", 
    "Controller",
    "Autopilot",
    # Key functions
    "compute_thrust",
    "density_at_altitude",
    "wind_at_position",
]