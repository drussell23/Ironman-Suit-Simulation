# File: backend/aerodynamics/flight_models/flight_dynamics.py

"""
flight_dynamics.py

Basic 3-DOF flight-dynamics model for the Iron Man suit:
- Computes lift & drag using angle of attack and altitude-dependent density
- Integrates motion in 6 states [x, y, z, vx, vy, vz] using RK4 for stability
"""
import numpy as np
from backend.aerodynamics.environmental_effects.atmospheric_density import (
    density_at_altitude,
)


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
        :param mass: suit mass (kg)
        :param wing_area: reference area for lift/drag (m^2)
        :param Cl0: lift coefficient at zero angle of attack
        :param Cld_alpha: lift slope per radian
        :param Cd0: drag coefficient at zero lift
        :param k: induced drag factor
        :param gravity: gravitational acceleration (m/s^2)
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
        Compute lift (Cl) and drag (Cd) coefficients for given AoA.
        """
        Cl = self.Cl0 + self.Cld_alpha * alpha
        Cd = self.Cd0 + self.k * Cl**2
        return Cl, Cd

    def aerodynamic_forces(
        self, velocity: np.ndarray, alpha: float, altitude: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute lift and drag forces in world frame.
        :param velocity: [vx, vy, vz] m/s
        :param alpha: angle of attack in radians
        :param altitude: height above sea level in meters
        :returns: (lift_vector, drag_vector)
        """
        V = np.linalg.norm(velocity)
        if V < 1e-6:
            return np.zeros(3), np.zeros(3)

        rho = density_at_altitude(altitude)
        Cl, Cd = self.aerodynamic_coeffs(alpha)
        q = 0.5 * rho * V**2

        # Lift acts upward (Y-axis)
        lift_mag = q * self.S * Cl
        lift = np.array([0.0, lift_mag, 0.0])

        # Drag opposes motion
        drag_mag = q * self.S * Cd
        drag = -drag_mag * (velocity / V)

        return lift, drag

    def derivatives(self, state: np.ndarray, control: dict) -> np.ndarray:
        """
        Compute time-derivatives [dx, dy, dz, dvx, dvy, dvz] of the state.
        :param state: [x, y, z, vx, vy, vz]
        :param control: { 'thrust': N, 'alpha': rad }
        :returns: derivatives as a 6-element numpy array
        """
        x, y, z, vx, vy, vz = state
        velocity = np.array([vx, vy, vz])
        thrust = control.get("thrust", 0.0)
        alpha = control.get("alpha", 0.0)

        # Aerodynamics
        lift, drag = self.aerodynamic_forces(velocity, alpha, y)

        # Thrust vector (aligned with up direction)
        thrust_vec = np.array([0.0, thrust / self.mass, 0.0])

        # Gravity force
        gravity_force = np.array([0.0, -self.g, 0.0]) * self.mass

        # Net acceleration
        accel = (lift + drag + thrust_vec + gravity_force) / self.mass

        return np.array([vx, vy, vz, *accel])

    def step(self, state: np.ndarray, control: dict, dt: float) -> np.ndarray:
        """
        Advance the state by dt using a 4th-order Runge-Kutta integrator.
        :param state: current state vector
        :param control: control dictionary
        :param dt: time-step (s)
        :returns: new state vector after dt
        """
        k1 = self.derivatives(state, control)
        k2 = self.derivatives(state + 0.5 * dt * k1, control)
        k3 = self.derivatives(state + 0.5 * dt * k2, control)
        k4 = self.derivatives(state + dt * k3, control)

        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
