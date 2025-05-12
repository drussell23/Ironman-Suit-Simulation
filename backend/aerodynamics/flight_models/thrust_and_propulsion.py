# backend/aerodynamics/flight_models/thrust_and_propulsion.py

"""
thrust_and_propulsion.py

Provides a simple thrust calculation for the Iron Man suit:

    Thrust (N) = mass (kg) x (desired_acceleration (m/s²) + gravity (m/s²))

By including gravity compensation, a positive 'acceleration' of zero will still produce enough thrust to hover. 
"""
def compute_thrust(mass: float, acceleration: float, gravity: float = 9.81) -> float:
    """
    Compute the thrust force required for a given mass and desired acceleration.

    :param mass: mass of the suit (kg)
    :param acceleration: desired additional acceleration (m/s²), positive upward
    :param gravity: gravitational acceleration (m/s²), default 9.81
    :return: thrust force in Newtons
    """
    # Total acceleration needed is acceleration + gravity
    total_accel = acceleration + gravity
    return mass * total_accel