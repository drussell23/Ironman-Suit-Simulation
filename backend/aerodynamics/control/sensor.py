import numpy as np
import logging
from typing import Any, Dict, Optional, Callable

from .autopilot import SensorError

logger = logging.getLogger(__name__)

class Sensor:
    """
    Sensor subsystem reading raw states, adding noise, and providing structured output.
    """
    def __init__(
        self,
        sensor_name: str = "sensor",
        state_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        noise_std: float = 0.0
    ):
        """
        Args:
            sensor_name: Identifier for sensor instance.
            state_provider: Callable returning a dict with keys:
                'position', 'velocity', 'angular_velocity', optionally 'acceleration', 'angular_acceleration'.
            noise_std: Standard deviation for Gaussian noise added to measurements.
        """
        self.sensor_name = sensor_name
        self.state_provider = state_provider
        self.noise_std = noise_std
        self._last_raw: Optional[Dict[str, Any]] = None

    def read_state(self) -> Dict[str, Any]:
        """
        Reads raw state, adds noise, and returns:
          - 'position', 'velocity', 'angular_velocity', 'acceleration', 'angular_acceleration'
        Raises:
            SensorError if provider missing or data invalid.
        """
        if self.state_provider is None:
            raise SensorError(
                message="No state_provider configured",
                sensor_name=self.sensor_name
            )
        try:
            raw = self.state_provider()
            self._last_raw = raw
            if not isinstance(raw, dict):
                raise SensorError(
                    message="state_provider returned invalid type",
                    sensor_name=self.sensor_name,
                    raw_data=raw
                )
            # Validate required keys
            for key in ("position", "velocity", "angular_velocity"):
                if key not in raw:
                    raise SensorError(
                        message=f"Missing '{key}' in sensor data",
                        sensor_name=self.sensor_name,
                        raw_data=raw
                    )
            # Convert to numpy arrays
            pos = np.array(raw["position"], float)
            vel = np.array(raw["velocity"], float)
            omega = np.array(raw["angular_velocity"], float)
            acc = np.array(raw.get("acceleration", np.zeros(3)), float)
            alpha = np.array(raw.get("angular_acceleration", np.zeros(3)), float)
            # Add Gaussian noise
            if self.noise_std > 0:
                pos += np.random.normal(0, self.noise_std, pos.shape)
                vel += np.random.normal(0, self.noise_std, vel.shape)
                omega += np.random.normal(0, self.noise_std, omega.shape)
                acc += np.random.normal(0, self.noise_std, acc.shape)
                alpha += np.random.normal(0, self.noise_std, alpha.shape)
            return {
                "position": pos,
                "velocity": vel,
                "angular_velocity": omega,
                "acceleration": acc,
                "angular_acceleration": alpha
            }
        except SensorError:
            raise
        except Exception as e:
            logger.error(f"Sensor.read_state exception: {e}")
            raise SensorError(
                message=f"Exception during read_state: {e}",
                sensor_name=self.sensor_name,
                raw_data=self._last_raw
            )