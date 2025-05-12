"""
_summary_
Aerodynamics package for Iron Man Suit Simulation.
Organizes environmental effects, and flight models modules.
"""

# Make subpackages available at package level.
from . import environmental_effects
from . import flight_models

__all__ = [
    "environmental_effects",
    "flight_models",
]