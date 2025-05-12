# Package initialization for environmental_effects

"""_summary_
Environmental effects package:
- atmospheric_density for altitude-dependent air density
- wind_interaction for position-based wind field simulation
"""
from .atmospheric_density import density_at_altitude
from .wind_interaction import wind_at_position

__all__ = [
    "density_at_altitude",
    "wind_at_position",
]
