# backend/aerodynamics/flight_models/thrust_and_propulsion.py

"""
thrust_and_propulsion.py

Provides a simple thrust calculation for the Iron Man suit:

    Thrust (N) = mass (kg) x (desired_acceleration (m/s²) + gravity (m/s²))

By including gravity compensation, a positive 'acceleration' of zero will still produce enough thrust to hover. 
"""
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def compute_thrust(
    mass: float,
    acceleration: float,
    gravity: float = 9.81,
    margin: float = 1.0,
    min_thrust: float = 0.0,
    max_thrust: Optional[float] = None
) -> float:
    """
    Compute the thrust force required for a given mass and desired acceleration.

    :param mass: mass of the suit (kg)
    :param acceleration: desired additional acceleration (m/s²), positive upward
    :param gravity: gravitational acceleration (m/s²), default 9.81
    :param margin: safety margin for thrust calculation, default 1.0
    :param min_thrust: minimum allowed thrust, default 0.0
    :param max_thrust: maximum allowed thrust, default None
    :return: thrust force in Newtons
    """
    # Total acceleration needed is acceleration + gravity
    total_accel = acceleration + gravity

    # Input validation
    if mass <= 0:
        raise ValueError("Mass must be positive.")
    if margin <= 0:
        raise ValueError("Margin must be positive.")
    if min_thrust < 0:
        raise ValueError("min_thrust must be non-negative.")
    if max_thrust is not None and max_thrust < min_thrust:
        raise ValueError("max_thrust must be >= min_thrust.")

    # Compute thrust with margin
    thrust = mass * total_accel * margin

    # Clamp thrust
    thrust = max(thrust, min_thrust)
    if max_thrust is not None:
        thrust = min(thrust, max_thrust)

    logger.debug(
        f"compute_thrust(mass={mass}, accel={acceleration}, gravity={gravity}, margin={margin}) -> thrust={thrust}"
    )

    return thrust