import numpy as np
import math
import logging

from aerodynamics.environmental_effects.atmospheric_density import density_at_altitude
from aerodynamics.environmental_effects.thermal_effects import temperature_at_altitude

logger = logging.getLogger(__name__)

# Constants for compressibility and atmosphere
GAS_CONSTANT = 287.05  # J/(kg·K)
GAMMA = 1.4  # ratio of specific heats

def compute_lift_coefficient(Cl0: float, Cld_alpha: float, alpha: float, stall_angle: float = np.deg2rad(15)) -> float:
    """
    Compute lift coefficient for a given angle of attack with stall saturation.

    :param Cl0: lift coefficient at zero AoA
    :param Cld_alpha: lift slope per rad
    :param alpha: angle of attack (rad)
    :param stall_angle: stall angle (rad) beyond which lift saturates
    :return: lift coefficient Cl
    """
    # Stall saturation on alpha
    if abs(alpha) > stall_angle:
        eff_alpha = stall_angle * np.sign(alpha)
    else:
        eff_alpha = alpha
    Cl_linear = Cl0 + Cld_alpha * eff_alpha
    logger.debug(f"Cl (alpha={alpha:.3f} rad) -> {Cl_linear:.3f}")
    return Cl_linear

def compute_drag_coefficient(Cd0: float, k: float, Cl: float) -> float:
    """
    Compute drag coefficient given lift coefficient.

    :param Cd0: zero-lift drag coefficient
    :param k: induced drag factor
    :param Cl: lift coefficient
    :return: drag coefficient Cd
    """
    Cd_val = Cd0 + k * Cl ** 2
    logger.debug(f"Cd (Cl={Cl:.3f}) -> {Cd_val:.3f}")
    return Cd_val

def aerodynamic_forces(
    velocity: np.ndarray,
    alpha: float,
    altitude: float,
    wing_area: float,
    Cl0: float,
    Cld_alpha: float,
    Cd0: float,
    k: float,
    stall_angle: float = np.deg2rad(15),
    compressibility: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute lift and drag forces in world frame with advanced aerodynamic effects.

    :param velocity: velocity vector [vx, vy, vz] in m/s
    :param alpha: angle of attack in radians
    :param altitude: height above sea level in meters
    :param wing_area: reference area for lift/drag in m^2
    :param Cl0: lift coefficient at zero AoA
    :param Cld_alpha: lift coefficient slope per rad
    :param Cd0: zero-lift drag coefficient
    :param k: induced drag factor
    :param stall_angle: stall angle (rad)
    :param compressibility: apply Prandtl-Glauert correction if True
    :return: (lift_vector, drag_vector) as numpy arrays
    """
    # Input validation
    velocity = np.asarray(velocity, dtype=float)
    if velocity.shape != (3,):
        raise ValueError("velocity must be a 3-element vector")
    for nm, val in [("alpha", alpha), ("altitude", altitude), ("wing_area", wing_area),
                   ("Cl0", Cl0), ("Cld_alpha", Cld_alpha), ("Cd0", Cd0), ("k", k)]:
        if not isinstance(val, (int, float)):
            raise TypeError(f"{nm} must be a numeric type")
    altitude = max(0.0, float(altitude))
    if wing_area <= 0:
        raise ValueError("wing_area must be positive")

    V = np.linalg.norm(velocity)
    if V < 1e-6:
        return np.zeros(3), np.zeros(3)

    # Atmospheric properties
    rho = density_at_altitude(altitude)
    # Lift coefficient with stall
    Cl = compute_lift_coefficient(Cl0, Cld_alpha, alpha, stall_angle)
    # Compressibility correction
    if compressibility:
        T = temperature_at_altitude(altitude)
        a = math.sqrt(GAMMA * GAS_CONSTANT * T)
        M = V / a
        if M < 1.0:
            beta = math.sqrt(1 - M**2)
            Cl /= beta
            logger.debug(f"Compressibility M={M:.3f}, beta={beta:.3f}, Cl corrected to {Cl:.3f}")
        else:
            logger.warning(f"Mach {M:.3f} >= 1.0, skipping compressibility correction")
    # Drag coefficient
    Cd = compute_drag_coefficient(Cd0, k, Cl)

    q = 0.5 * rho * V ** 2
    # Lift and drag vectors
    # Lift acts upward (y-axis)
    lift_mag = q * wing_area * Cl
    lift = np.array([0.0, lift_mag, 0.0])

    # Drag opposes motion
    drag_mag = q * wing_area * Cd
    drag = -drag_mag * (velocity / V)

    logger.debug(f"aerodynamic_forces: V={V:.2f}, alt={altitude:.1f}, Cl={Cl:.3f}, Cd={Cd:.3f}, lift={lift}, drag={drag}")
    return lift, drag