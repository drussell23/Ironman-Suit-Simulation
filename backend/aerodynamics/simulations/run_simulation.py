#!/usr/bin/env python3
"""
Main simulation runner for Iron Man suit flight dynamics.
Integrates aerodynamics, control systems, and provides visualization.
"""

import numpy as np
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, Any

from ..flight_models.flight_dynamics import FlightDynamics
from ..control.autopilot import Autopilot
from ..environmental_effects.atmospheric_density import density_at_altitude

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IronManSimulation:
    def __init__(self, config_path: str = None):
        """Initialize the Iron Man suit simulation."""
        self.config = self._load_config(config_path)
        self.flight_dynamics = self._setup_flight_dynamics()
        self.autopilot = self._setup_autopilot()
        self.time = 0.0
        self.dt = self.config.get("simulation", {}).get("dt", 0.01)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load simulation configuration."""
        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                return yaml.safe_load(f)

        # Default configuration
        return {
            "suit": {
                "mass": 100.0,  # kg
                "wing_area": 2.0,  # m^2
                "Cl0": 0.1,
                "Cld_alpha": 5.0,
                "Cd0": 0.02,
                "k": 0.1,
            },
            "simulation": {
                "dt": 0.01,  # seconds
                "duration": 10.0,  # seconds
                "initial_state": [
                    0.0,
                    100.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],  # [x, y, z, vx, vy, vz]
            },
            "control": {
                "target_altitude": 150.0,  # meters
                "max_thrust": 2000.0,  # Newtons
                "kp": 1.0,
                "ki": 0.1,
                "kd": 0.5,
            },
        }

    def _setup_flight_dynamics(self) -> FlightDynamics:
        """Initialize flight dynamics model."""
        suit_config = self.config["suit"]
        return FlightDynamics(
            mass=suit_config["mass"],
            wing_area=suit_config["wing_area"],
            Cl0=suit_config["Cl0"],
            Cld_alpha=suit_config["Cld_alpha"],
            Cd0=suit_config["Cd0"],
            k=suit_config["k"],
        )

    def _setup_autopilot(self) -> Autopilot:
        """Initialize autopilot system."""
        # This would integrate with your existing autopilot
        # For now, return a simple controller
        return None  # Placeholder

    def run_simulation(self) -> Dict[str, np.ndarray]:
        """Run the complete simulation."""
        logger.info("Starting Iron Man suit simulation...")

        # Initialize state
        state = np.array(self.config["simulation"]["initial_state"])
        duration = self.config["simulation"]["duration"]
        dt = self.config["simulation"]["dt"]

        # Storage for results
        times = []
        states = []
        controls = []

        # Simulation loop
        while self.time <= duration:
            # Store current state
            times.append(self.time)
            states.append(state.copy())

            # Simple hover control (placeholder)
            control = self._compute_control(state)
            controls.append(control)

            # Integrate dynamics
            state = self.flight_dynamics.step(state, control, dt)
            self.time += dt

            # Log progress
            if int(self.time * 10) % 10 == 0:
                logger.info(
                    f"Time: {self.time:.1f}s, Altitude: {state[1]:.1f}m, "
                    f"Velocity: {np.linalg.norm(state[3:6]):.1f} m/s"
                )

        logger.info("Simulation completed successfully!")

        return {
            "times": np.array(times),
            "states": np.array(states),
            "controls": np.array(controls),
        }

    def _compute_control(self, state: np.ndarray) -> Dict[str, float]:
        """Compute control inputs based on current state."""
        # Simple PID controller for altitude hold
        target_altitude = self.config["control"]["target_altitude"]
        current_altitude = state[1]
        altitude_error = target_altitude - current_altitude

        # Simple proportional control
        kp = self.config["control"]["kp"]
        thrust = kp * altitude_error

        # Clamp thrust
        max_thrust = self.config["control"]["max_thrust"]
        thrust = np.clip(thrust, 0, max_thrust)

        return {"thrust": thrust, "alpha": 0.0}  # Zero angle of attack for hover

    def plot_results(self, results: Dict[str, np.ndarray]):
        """Plot simulation results."""
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))

            # Altitude over time
            axes[0, 0].plot(results["times"], results["states"][:, 1])
            axes[0, 0].set_ylabel("Altitude (m)")
            axes[0, 0].set_xlabel("Time (s)")
            axes[0, 0].grid(True)

            # Velocity over time
            velocities = np.linalg.norm(results["states"][:, 3:6], axis=1)
            axes[0, 1].plot(results["times"], velocities)
            axes[0, 1].set_ylabel("Velocity (m/s)")
            axes[0, 1].set_xlabel("Time (s)")
            axes[0, 1].grid(True)

            # Thrust over time
            thrusts = [c["thrust"] for c in results["controls"]]
            axes[1, 0].plot(results["times"], thrusts)
            axes[1, 0].set_ylabel("Thrust (N)")
            axes[1, 0].set_xlabel("Time (s)")
            axes[1, 0].grid(True)

            # 3D trajectory
            axes[1, 1].plot(
                results["states"][:, 0],
                results["states"][:, 2],
                results["states"][:, 1],
            )
            axes[1, 1].set_xlabel("X (m)")
            axes[1, 1].set_ylabel("Z (m)")
            axes[1, 1].set_zlabel("Y (Altitude) (m)")
            axes[1, 1].grid(True)

            plt.tight_layout()
            plt.savefig("ironman_simulation_results.png", dpi=300, bbox_inches="tight")
            plt.show()

        except ImportError:
            logger.warning("matplotlib not available. Skipping plotting.")
            logger.info("Simulation results saved in results dictionary.")


def main():
    parser = argparse.ArgumentParser(description="Iron Man Suit Simulation")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--plot", action="store_true", help="Plot results")

    args = parser.parse_args()

    # Run simulation
    sim = IronManSimulation(args.config)
    results = sim.run_simulation()

    # Plot if requested
    if args.plot:
        sim.plot_results(results)

    logger.info("Simulation completed!")


if __name__ == "__main__":
    main()


