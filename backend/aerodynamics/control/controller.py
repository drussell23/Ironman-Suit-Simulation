import numpy as np
import logging
from typing import Any, Dict, Optional

from .autopilot import ControllerError

logger = logging.getLogger(__name__)

class Controller:
    """
    PD controller for 6-DOF vehicle:
      - translation: force = m*(Kp*vel_error + Kd*acc_error)
      - rotation:  moment = I*(Kp*omega_error + Kd*alpha_error)

    Args:
        mass: vehicle mass
        inertia: 3×3 inertia matrix
        kp_trans: proportional gain for translation
        kd_trans: derivative gain for translation
        kp_rot: proportional gain for rotation
        kd_rot: derivative gain for rotation
    """
    def __init__(
        self,
        mass: float = 1.0,
        inertia: Optional[np.ndarray] = None,
        kp_trans: float = 1.0,
        kd_trans: float = 0.0,
        kp_rot: float = 1.0,
        kd_rot: float = 0.0
    ):
        self.mass = mass
        self.inertia = inertia if inertia is not None else np.eye(3)
        self.kp_trans = kp_trans
        self.kd_trans = kd_trans
        self.kp_rot = kp_rot
        self.kd_rot = kd_rot
        self._prev_vel_error = np.zeros(3)
        self._prev_omega_error = np.zeros(3)

    def compute_control(
        self,
        state: Dict[str, Any],
        reference: Dict[str, Any],
        dt: float
    ) -> Dict[str, np.ndarray]:
        """
        Compute force and moment commands.

        state expects keys:
          - 'velocity': (3,) array
          - 'angular_velocity': (3,) array
          - optional 'acceleration', 'angular_acceleration'
        reference expects keys:
          - 'velocity_cmd': (3,) array
          - 'angular_velocity_cmd': (3,) array

        Raises:
            ControllerError on missing keys or computation errors.
        """
        try:
            vel = np.array(state['velocity'], float)
            omega = np.array(state['angular_velocity'], float)
            acc = np.array(state.get('acceleration', np.zeros(3)), float)
            alpha = np.array(state.get('angular_acceleration', np.zeros(3)), float)

            vel_ref = np.array(reference['velocity_cmd'], float)
            omega_ref = np.array(reference['angular_velocity_cmd'], float)
        except Exception as e:
            raise ControllerError(
                message=f"Controller missing state/reference data: {e}",
                state=state,
                reference=reference,
                dt=dt
            )

        # Translation PD
        vel_error = vel_ref - vel
        d_vel_error = (vel_error - self._prev_vel_error) / dt if dt > 0 else np.zeros(3)
        force_cmd = self.mass * (self.kp_trans * vel_error + self.kd_trans * d_vel_error)
        self._prev_vel_error = vel_error

        # Rotation PD
        omega_error = omega_ref - omega
        d_omega_error = (omega_error - self._prev_omega_error) / dt if dt > 0 else np.zeros(3)
        moment_cmd = self.inertia.dot(self.kp_rot * omega_error + self.kd_rot * d_omega_error)
        self._prev_omega_error = omega_error

        logger.debug(
            f"compute_control dt={dt:.6f}, force_cmd={force_cmd}, moment_cmd={moment_cmd}"
        )

        return {'force_cmd': force_cmd, 'moment_cmd': moment_cmd}