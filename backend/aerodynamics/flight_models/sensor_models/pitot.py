import numpy as np
import math
import logging

from aerodynamics.environmental_effects.atmospheric_density import density_at_altitude
from aerodynamics.environmental_effects.thermal_effects import temperature_at_altitude

logger = logging.getLogger(__name__)

SEA_LEVEL_DENSITY = 1.225  # kg/m^3
GAS_CONSTANT = 287.05     # J/(kg·K)
GAMMA = 1.4               # specific heat ratio

class PitotSensor:
    """
    Pitot tube sensor model: measures static and total pressure to infer airspeeds.
    Supports optional noise, bias, and compressibility corrections.
    """

    def __init__(
        self,
        noise_std_static: float = 0.0,
        noise_std_total: float = 0.0,
        bias_static: float = 0.0,
        bias_total: float = 0.0,
    ):
        self.noise_std_static = noise_std_static
        self.noise_std_total = noise_std_total
        self.bias_static = bias_static
        self.bias_total = bias_total

    def measure_static_pressure(self, altitude: float) -> float:
        """
        Simulate static pressure measurement (Pa) at given altitude.
        """
        alt = float(altitude)
        if alt < 0:
            alt = 0.0
        rho = density_at_altitude(alt)
        T = temperature_at_altitude(alt)
        p_static = rho * GAS_CONSTANT * T
        noise = np.random.normal(0.0, self.noise_std_static)
        meas = p_static + noise + self.bias_static
        logger.debug(f"Static pressure: true={p_static:.2f} Pa, meas={meas:.2f} Pa")
        return meas

    def measure_total_pressure(self, velocity: np.ndarray, altitude: float) -> float:
        """
        Simulate total (stagnation) pressure (Pa) for given airspeed and altitude.
        """
        vel = np.asarray(velocity, dtype=float)
        if vel.shape != (3,):
            raise ValueError("velocity must be a 3-element vector")
        V = np.linalg.norm(vel)
        alt = float(altitude)
        if alt < 0:
            alt = 0.0
        rho = density_at_altitude(alt)
        T = temperature_at_altitude(alt)
        p_static = rho * GAS_CONSTANT * T
        q_dyn = 0.5 * rho * V**2
        p_total = p_static + q_dyn
        noise = np.random.normal(0.0, self.noise_std_total)
        meas = p_total + noise + self.bias_total
        logger.debug(f"Total pressure: true={p_total:.2f} Pa, meas={meas:.2f} Pa")
        return meas

    def measure_dynamic_pressure(self, velocity: np.ndarray, altitude: float) -> float:
        """
        Differential pressure by pitot: total - static.
        """
        p_t = self.measure_total_pressure(velocity, altitude)
        p_s = self.measure_static_pressure(altitude)
        q = p_t - p_s
        logger.debug(f"Dynamic pressure: {q:.2f} Pa")
        return q

    def indicated_airspeed(self, q: float) -> float:
        """
        Indicated Airspeed (IAS) from dynamic pressure, using sea-level density.
        """
        qm = max(0.0, float(q))
        v_ias = math.sqrt(2 * qm / SEA_LEVEL_DENSITY)
        logger.debug(f"IAS: q={qm:.2f} Pa -> v_ias={v_ias:.2f} m/s")
        return v_ias

    def true_airspeed(self, q: float, altitude: float, compressibility: bool = False) -> float:
        """
        True Airspeed (TAS) from dynamic pressure at altitude.
        Optionally apply compressibility correction.
        """
        qm = max(0.0, float(q))
        alt = float(altitude)
        if alt < 0:
            alt = 0.0
        rho = density_at_altitude(alt)
        if rho <= 0:
            raise ValueError("Invalid density for altitude")
        v = math.sqrt(2 * qm / rho)
        if compressibility:
            p_static = rho * GAS_CONSTANT * temperature_at_altitude(alt)
            pr = qm / p_static + 1.0
            M = math.sqrt((2 / (GAMMA - 1)) * (pr**((GAMMA - 1)/GAMMA) - 1.0))
            a = math.sqrt(GAMMA * GAS_CONSTANT * temperature_at_altitude(alt))
            v = M * a
            logger.debug(f"Compressibility: M={M:.3f}, TAS corrected={v:.2f} m/s")
        logger.debug(f"TAS: q={qm:.2f} Pa, alt={alt:.1f} m -> v_tas={v:.2f} m/s")
        return v