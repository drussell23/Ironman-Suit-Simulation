import numpy as np
import logging
from typing import Any, Dict

from backend.autopilot.waypoint_planner import WaypointPlanner
from .autopilot import GuidanceError

logger = logging.getLogger(__name__)

class Guidance:
    """
    Guidance subsystem generating setpoints via a WaypointPlanner.
    """
    def __init__(
        self,
        waypoints: list,
        speed: float = 1.0,
        acceptance_radius: float = 0.5
    ):
        """
        Args:
            waypoints: list of [x, y, z] waypoints.
            speed: desired speed magnitude (m/s).
            acceptance_radius: radius (m) to consider waypoint reached.
        """
        self.planner = WaypointPlanner(waypoints, acceptance_radius)
        self.speed = speed

    def get_reference(self, state: Any, dt: float) -> Dict[str, Any]:
        """
        Compute the next reference for controller.

        Args:
            state: dict containing 'position' key with (3,) value.
            dt: time delta since last call.

        Returns:
            A dict with 'velocity_cmd' and 'angular_velocity_cmd' (both np.ndarray).
        Raises:
            GuidanceError on missing data or compute failure.
        """
        try:
            pos = np.array(state['position'], float)
            # advance planner if close to current waypoint
            self.planner.update(pos)
            vel_cmd = self.planner.get_desired_velocity(pos, self.speed)
            omega_cmd = np.zeros(3)  # no rotation command
            return {'velocity_cmd': vel_cmd, 'angular_velocity_cmd': omega_cmd}
        except Exception as e:
            logger.error(f"Guidance.get_reference failed: {e}")
            raise GuidanceError(
                message=f"Guidance failure: {e}",
                state=state,
                dt=dt,
                reference=None
            )