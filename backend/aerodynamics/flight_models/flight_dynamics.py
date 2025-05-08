# File: aerodynamics/flight_models/flight_dynamics.py

"""
flight_dynamics.py

Basic flight-dynamics model for the Iron Man suit:
- Computes lift & drag from velocity, angle of attack, and altitude
- Integrates motion in 3 DOF (x, z, and vertical y)
- RK4 stepper for stable simulation
"""

import numpy as np
from environmental_effects.atmospheric_density import density_at_altitude
from environmental_effects.wind_interaction import wind_at_position

class FlightDynamics:
    def __init__(
        self,
        mass: float,
        wing_area: float,
        Cl0: float,
        Cld_alpha: float,
        Cd0: float,
        k: float,
        gravity: float = 9.81,
    ):
        """
        Args:
            mass (float): mass of the suit (kg)
            wing_area (float): reference area for lift/drag (m^2)
            Cl0 (float): zero-AoA lift coefficient
            Cld_alpha (float): lift slope (per radian)
            Cd0 (float): zero-lift drag coefficient
            k (float): induced drag factor
            gravity (float, optional): _description_. Defaults to 9.81.
        """
        self.mass = mass
        self.S = wing_area
        self.Cl0 = Cl0
        self.Cld_alpha = Cld_alpha
        self.Cd0 = Cd0
        self.k = k
        self.g = gravity

    def aerodynamic_coeffs(self, alpha: float) -> tuple[float, float]:
        """
        Compute lift and drag coefficients given angle of attack.
        :returns: (Cl, Cd)
        """
        Cl = self.Cl0 + self.Cld_alpha * alpha
        Cd = self.Cd0 + self.k * Cl**2

        return Cl, Cd

    def aerodynamic_forces(
        self, velocity: np.ndarray, alpha: float, altitude: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute lift and drag forces in the body‐fixed frame.
        :param velocity: 3D velocity vector (m/s)
        :param alpha: angle of attack (rad)
        :param altitude: height above sea level (m)
        :returns: (lift_vector, drag_vector)
        """
        V = np.linalg.norm(velocity)

        if V < 1e-3:
            return np.zeros(3), np.zeros(3)

        rho = density_at_altitude(altitude)
        Cl, Cd = self.aerodynamic_coeffs(alpha)
        q = 0.5 * rho * V**2

        # Lift acts perpendicular to velocity in the vertical plane.
        lift_mag = q * self.S * Cl
        lift_dir = np.array([0, 1, 0])  # Simplistic: Straight up.
        lift = lift_mag * lift_dir

        # Drag acts opposite to the velocity vector.
        drag_mag = q * self.S * Cd
        drag = -drag_mag * (velocity / V)

        return lift, drag
    
    def derivatives(self, state: np.ndarray, control: dict) -> np.ndarray:
        # Unpack, now actually using x and z
        x, y, z, vx, vy, vz = state
        velocity = np.array([vx, vy, vz])

        # Get local wind at this x,z position
        wind = wind_at_position(x, z)

        # Compute relative velocity (airflow relative to the suit)
        rel_vel = velocity - wind

        thrust = control.get("thrust", 0.0)
        alpha  = control.get("alpha",  0.0)

        # Aerodynamic forces now use rel_vel instead of raw velocity
        lift, drag = self.aerodynamic_forces(rel_vel, alpha, y)

        # Thrust (aligned with local up)
        thrust_vec = np.array([0, thrust / self.mass, 0])

        # Gravity
        gravity_force = np.array([0, -self.g, 0]) * self.mass

        # Net acceleration
        accel = (lift + drag + thrust_vec + gravity_force) / self.mass

        # Derivative: [dx/dt, dy/dt, dz/dt, dvx/dt, dvy/dt, dvz/dt]
        return np.array([vx, vy, vz, *accel])

    
    
