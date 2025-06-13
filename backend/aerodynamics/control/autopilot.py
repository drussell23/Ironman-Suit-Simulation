# File: backend/aerodynamics/control/autopilot.py

import time 
import logging 
from typing import Any, Dict, List, Optional, Type, Union, Callable 

# Import the four subsystems. Adjust these imports if your file locations differ. 
from .sensor import Sensor 
from .guidance import Guidance 
from .controller import Controller 
from .actuator import Actuator 


# ----------------------------------------------------------------------
# Custom Exceptions
# ----------------------------------------------------------------------

class AutopilotError(Exception):
    def __init__(
        self,
        message: str = "",
        code: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            message: Human-readable error message.
            code: Numeric error code.
            context: Arbitrary key/value pairs with debug info.
        """
        super().__init__(message)
        self.message = message
        self.code = code
        self.context = context or {}

    def __str__(self) -> str:
        """
        Return formatted string with code and any context entries.
        """
        parts = []
        if self.code is not None:
            parts.append(f"[Error {self.code}]")
        parts.append(self.message or super().__str__())
        for key, value in self.context.items():
            parts.append(f"({key}={repr(value)})")
        return " ".join(parts)


class SensorError(AutopilotError):
    def __init__(
        self,
        message: str = "Sensor subsystem error",
        code: Optional[int] = None,
        *,
        sensor_name: Optional[str] = None,
        raw_data: Optional[Any] = None,
        state: Optional[Any] = None
    ):
        super().__init__(message, code)
        self.sensor_name = sensor_name
        self.raw_data = raw_data
        self.state = state

    def __str__(self) -> str:
        parts = []
        if self.code is not None:
            parts.append(f"[Code {self.code}]")
        parts.append(self.message)
        if self.sensor_name is not None:
            parts.append(f"(sensor={self.sensor_name})")
        if self.raw_data is not None:
            parts.append(f"(raw_data={repr(self.raw_data)})")
        if self.state is not None:
            parts.append(f"(state={repr(self.state)})")
        return " ".join(parts)


class GuidanceError(AutopilotError):
    def __init__(
        self,
        message: str = "Guidance subsystem error",
        code: Optional[int] = None,
        *,
        state: Optional[Any] = None,
        dt: Optional[float] = None,
        reference: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, code)
        self.state = state
        self.dt = dt
        self.reference = reference

    def __str__(self) -> str:
        parts = []
        if self.code is not None:
            parts.append(f"[Code {self.code}]")
        parts.append(self.message)

        # Append context if available
        if self.state is not None:
            parts.append(f"(state={repr(self.state)})")
        if self.dt is not None:
            parts.append(f"(dt={self.dt:.6f})")
        if self.reference is not None:
            parts.append(f"(reference={repr(self.reference)})")

        return " ".join(parts)


class ControllerError(AutopilotError):
    def __init__(
        self,
        message: str = "Controller subsystem error",
        code: Optional[int] = None,
        *,
        state: Optional[Any] = None,
        reference: Optional[Dict[str, Any]] = None,
        dt: Optional[float] = None,
        control: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, code)
        self.state = state
        self.reference = reference
        self.dt = dt
        self.control = control

    def __str__(self) -> str:
        """
        Returns a detailed string including message, code, and any captured context.
        """
        parts = []
        if self.code is not None:
            parts.append(f"[Code {self.code}]")
        parts.append(self.message)

        # Append context if available
        if self.state is not None:
            parts.append(f"(state={repr(self.state)})")
        if self.reference is not None:
            parts.append(f"(reference={repr(self.reference)})")
        if self.control is not None:
            parts.append(f"(control={repr(self.control)})")
        if self.dt is not None:
            parts.append(f"(dt={self.dt:.6f})")

        return " ".join(parts)


class ActuatorError(AutopilotError):
    def __init__(
        self,
        message: str = "Actuator subsystem error",
        code: Optional[int] = None,
        *,
        state: Optional[Any] = None,
        reference: Optional[Dict[str, Any]] = None,
        control: Optional[Dict[str, Any]] = None,
        dt: Optional[float] = None
    ):
        super().__init__(message, code)
        self.state = state
        self.reference = reference
        self.control = control
        self.dt = dt

    def __str__(self) -> str:
        """
        Returns a detailed string including message, code, and any captured context.
        """
        parts = []
        if self.code is not None:
            parts.append(f"[Code {self.code}]")
        parts.append(self.message)

        # Append context if available
        if self.state is not None:
            parts.append(f"(state={repr(self.state)})")
        if self.reference is not None:
            parts.append(f"(reference={repr(self.reference)})")
        if self.control is not None:
            parts.append(f"(control={repr(self.control)})")
        if self.dt is not None:
            parts.append(f"(dt={self.dt:.6f})")

        return " ".join(parts)


# ----------------------------------------------------------------------
# Autopilot Class
# ----------------------------------------------------------------------

class Autopilot:
    def __init__(
        self,
        sensor: Sensor,
        guidance: Guidance,
        controller: Controller,
        actuator: Actuator,
        enable_logging: bool = True,
        enable_history: bool = False,
    ) -> None:
        # Validate injected subsystems
        if not isinstance(sensor, Sensor):
            raise TypeError(f"Expected Sensor instance, got {type(sensor)}")
        if not isinstance(guidance, Guidance):
            raise TypeError(f"Expected Guidance instance, got {type(guidance)}")
        if not isinstance(controller, Controller):
            raise TypeError(f"Expected Controller instance, got {type(controller)}")
        if not isinstance(actuator, Actuator):
            raise TypeError(f"Expected Actuator instance, got {type(actuator)}")

        self._sensor = sensor
        self._guidance = guidance
        self._controller = controller
        self._actuator = actuator

        self._last_time: Optional[float] = None
        self._stop_requested = False
        self._pause_requested = False

        # History stores a list of dicts, one per step
        self._history_enabled = enable_history
        self._history: List[Dict[str, Any]] = []

        # Configure a logger for this class
        self.logger = logging.getLogger(self.__class__.__name__)
        if enable_logging and not self.logger.hasHandlers():
            # Basic configuration if no handlers exist
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
                datefmt="%H:%M:%S",
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        elif not enable_logging:
            self.logger.disabled = True

        self.logger.info("Autopilot instance created.")
        if enable_history:
            self.logger.info("History tracking is ENABLED.")

    # ------------------------------------------------------------------
    # Properties & Setters for runtime swapping
    # ------------------------------------------------------------------

    @property
    def sensor(self) -> Sensor:
        return self._sensor

    @sensor.setter
    def sensor(self, new_sensor: Sensor) -> None:
        if not isinstance(new_sensor, Sensor):
            raise TypeError(f"Expected Sensor instance, got {type(new_sensor)}")
        self._sensor = new_sensor
        self.logger.info("Sensor subsystem has been swapped at runtime.")

    @property
    def guidance(self) -> Guidance:
        return self._guidance

    @guidance.setter
    def guidance(self, new_guidance: Guidance) -> None:
        if not isinstance(new_guidance, Guidance):
            raise TypeError(f"Expected Guidance instance, got {type(new_guidance)}")
        self._guidance = new_guidance
        self.logger.info("Guidance subsystem has been swapped at runtime.")

    @property
    def controller(self) -> Controller:
        return self._controller

    @controller.setter
    def controller(self, new_controller: Controller) -> None:
        if not isinstance(new_controller, Controller):
            raise TypeError(f"Expected Controller instance, got {type(new_controller)}")
        self._controller = new_controller
        self.logger.info("Controller subsystem has been swapped at runtime.")

    @property
    def actuator(self) -> Actuator:
        return self._actuator

    @actuator.setter
    def actuator(self, new_actuator: Actuator) -> None:
        if not isinstance(new_actuator, Actuator):
            raise TypeError(f"Expected Actuator instance, got {type(new_actuator)}")
        self._actuator = new_actuator
        self.logger.info("Actuator subsystem has been swapped at runtime.")

    # ------------------------------------------------------------------
    # History‐tracking methods
    # ------------------------------------------------------------------

    def enable_history(self, enabled: bool = True) -> None:
        """
        Turn history‐tracking on or off.

        If enabled, each call to step() will append a dictionary:
            { "timestamp": float, "dt": float, "state": Any, "reference": dict, "control": dict }
        to internal self._history.
        """
        self._history_enabled = bool(enabled)
        if self._history_enabled:
            self.logger.info("History tracking ENABLED.")
        else:
            self.logger.info("History tracking DISABLED.")
            self._history.clear()

    def clear_history(self) -> None:
        """Erase any stored history entries."""
        self._history.clear()
        self.logger.debug("History cleared.")

    def get_history(self) -> List[Dict[str, Any]]:
        """Return a shallow copy of the internal history list."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Pause / Resume / Stop methods
    # ------------------------------------------------------------------

    def pause(self) -> None:
        """
        Request a pause in the run() loop. The next iteration will break out and wait until resume() is called.
        """
        self._pause_requested = True
        self.logger.info("Pause requested.")

    def resume(self) -> None:
        """
        Resume a paused run() loop.
        """
        self._pause_requested = False
        self.logger.info("Resuming from pause.")

    def stop(self) -> None:
        """
        Request that the run() loop terminate early.
        """
        self._stop_requested = True
        self.logger.info("Stop requested.")

    # ------------------------------------------------------------------
    # Core step() method
    # ------------------------------------------------------------------

    def step(self, current_time: Optional[float] = None) -> Dict[str, Any]:
        now = current_time if (current_time is not None) else time.time()

        # 1) Compute dt
        if self._last_time is None:
            dt = 0.0
            self.logger.debug("First step() call; setting dt = 0.0")
        else:
            dt = now - self._last_time
            if dt < 0:
                self.logger.warning(
                    f"Negative dt detected ({dt:.6f}); clamping to 0."
                )
                dt = 0.0
        self._last_time = now

        # 2) Read the current state from the Sensor
        try:
            state = self._sensor.read_state()
            if state is None:
                raise SensorError("Sensor returned None for state.")
        except Exception as e:
            self.logger.error(f"Sensor.read_state() failed: {e}")
            raise SensorError(f"Sensor failure: {e}")

        # 3) Ask Guidance for the next setpoint
        try:
            reference = self._guidance.get_reference(state=state, dt=dt)
            if reference is None or not isinstance(reference, dict):
                raise GuidanceError("Guidance did not return a valid dict.")
        except Exception as e:
            self.logger.error(f"Guidance.get_reference() failed: {e}")
            raise GuidanceError(f"Guidance failure: {e}")

        # 4) Ask Controller for actuator commands
        try:
            control_commands = self._controller.compute_control(
                state=state, reference=reference, dt=dt
            )
            if control_commands is None or not isinstance(control_commands, dict):
                raise ControllerError("Controller did not return a valid dict.")
        except Exception as e:
            self.logger.error(f"Controller.compute_control() failed: {e}")
            raise ControllerError(f"Controller failure: {e}")

        # 5) Send commands to Actuator
        try:
            self._actuator.apply(control_commands)
        except Exception as e:
            self.logger.error(f"Actuator.apply() failed: {e}")
            raise ActuatorError(f"Actuator failure: {e}")

        # 6) Log details at DEBUG level
        self.logger.debug(
            f"step() → dt: {dt:.6f}, state: {state}, reference: {reference}, control: {control_commands}"
        )

        # 7) Store in history if enabled
        step_data = {
            "timestamp": now,
            "dt": dt,
            "state": state,
            "reference": reference,
            "control": control_commands,
        }
        if self._history_enabled:
            self._history.append(step_data)

        return step_data

    # ------------------------------------------------------------------
    # Main run() loop
    # ------------------------------------------------------------------

    def run(
        self,
        duration: float,
        rate_hz: float,
        log_every_n: Optional[int] = None,
        on_step_callback: Optional[Callable[[Dict[str, Any], int], None]] = None,
    ) -> None:
        if rate_hz <= 0:
            raise ValueError("rate_hz must be positive.")
        if duration <= 0:
            raise ValueError("duration must be positive.")

        self._stop_requested = False
        self._pause_requested = False

        period = 1.0 / rate_hz
        total_steps = int(duration * rate_hz)

        self.logger.info(
            f"Starting run: duration={duration:.3f}s, rate={rate_hz:.1f}Hz, "
            f"total_steps={total_steps}"
        )

        for i in range(total_steps):
            # 1) Check if a stop was requested
            if self._stop_requested:
                self.logger.info(f"Run stopped early at step {i}/{total_steps}.")
                break

            # 2) Check if we need to pause
            while self._pause_requested and not self._stop_requested:
                self.logger.info("Autopilot is paused. Sleeping for 0.1s …")
                time.sleep(0.1)
            if self._stop_requested:
                self.logger.info(f"Run stopped early at step {i}/{total_steps}.")
                break

            # 3) Perform one step
            t_start = time.time()
            try:
                step_data = self.step(current_time=t_start)
            except AutopilotError as e:
                self.logger.error(f"AutopilotError at step {i}: {e}")
                break  # on any step failure, exit run()

            # 4) Optionally log a summary
            if log_every_n and (i % log_every_n == 0):
                state = step_data["state"]
                reference = step_data["reference"]
                control = step_data["control"]
                self.logger.info(
                    f"[Step {i+1}/{total_steps}] dt={step_data['dt']:.4f}s, "
                    f"state={state}, reference={reference}, control={control}"
                )

            # 5) If the caller provided a callback, invoke it
            if on_step_callback:
                try:
                    on_step_callback(step_data, i)
                except Exception as e:
                    self.logger.warning(f"on_step_callback raised exception: {e}")

            # 6) Sleep until next iteration (account for computation time)
            t_elapsed = time.time() - t_start
            t_to_sleep = period - t_elapsed
            if t_to_sleep > 0:
                time.sleep(t_to_sleep)

        self.logger.info("Run completed or terminated.")

    # ------------------------------------------------------------------
    # Context‐manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "Autopilot":
        """
        Called when entering a `with … as auto:` block. Resets timing.
        """
        self._last_time = None
        self._stop_requested = False
        self._pause_requested = False
        if self._history_enabled:
            self._history.clear()
        self.logger.debug("__enter__(): Autopilot context initialized.")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        Called when exiting a `with` block. Attempt cleanup if needed.
        """
        # For now, just log. In a real system, you might close hardware connections, etc.
        self.logger.debug("__exit__(): Autopilot context exiting.")
        if exc_type:
            self.logger.error(f"Exception in context: {exc_type}, {exc_value}")
        # Do not suppress exceptions; let them propagate
        return False

    # ------------------------------------------------------------------
    # Utility: Reset internal timers and history
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Reset the autopilot’s internal timebase and history.
        Future calls to step() will treat the next iteration as dt=0.
        """
        self._last_time = None
        if self._history_enabled:
            self._history.clear()
        self.logger.info("Autopilot timers and history have been reset.")
