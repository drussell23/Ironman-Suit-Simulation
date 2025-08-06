#!/usr/bin/env python3
"""
Bare minimum startup script for Iron Man suit simulation.
This gets the essential components running for basic flight simulation.
"""

import sys
import os
import subprocess
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add backend to Python path
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

def check_requirements():
    """Check if essential Python packages are installed."""
    required_packages = ["numpy", "scipy", "yaml", "matplotlib"]
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        logger.error(f"Missing required packages: {', '.join(missing)}")
        logger.info("Install with: pip install numpy scipy pyyaml matplotlib")
        return False
    
    return True

def build_physics_plugin():
    """Build the C++ physics plugin if needed."""
    plugin_dir = REPO_ROOT / "backend" / "aerodynamics" / "physics_plugin"
    build_dir = plugin_dir / "build"
    
    # Check if already built
    lib_files = [
        build_dir / "libaerodynamics_physics_plugin.dylib",  # macOS
        build_dir / "libaerodynamics_physics_plugin.so",     # Linux
        build_dir / "aerodynamics_physics_plugin.dll"        # Windows
    ]
    
    if any(lib.exists() for lib in lib_files):
        logger.info("Physics plugin already built")
        return True
    
    logger.info("Building physics plugin...")
    
    # Create build directory
    build_dir.mkdir(exist_ok=True)
    
    # Run cmake
    try:
        subprocess.run(["cmake", ".."], cwd=build_dir, check=True)
        subprocess.run(["cmake", "--build", ".", "--", "-j4"], cwd=build_dir, check=True)
        logger.info("Physics plugin built successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to build physics plugin: {e}")
        logger.info("You can still run basic Python simulations without the plugin")
        return False

def create_default_config():
    """Create a default configuration file for the simulation."""
    config_path = REPO_ROOT / "simulation_config.yaml"
    
    if config_path.exists():
        logger.info(f"Using existing config: {config_path}")
        return str(config_path)
    
    config_content = """# Iron Man Suit Simulation Configuration
suit:
  mass: 100.0          # kg - Total suit + pilot mass
  wing_area: 2.0       # m^2 - Effective aerodynamic area
  Cl0: 0.1             # Base lift coefficient
  Cld_alpha: 5.0       # Lift curve slope
  Cd0: 0.02            # Base drag coefficient
  k: 0.1               # Induced drag factor

simulation:
  dt: 0.01             # seconds - Time step
  duration: 30.0       # seconds - Total simulation time
  initial_state:       # [x, y, z, vx, vy, vz]
    - 0.0              # x position (m)
    - 100.0            # y position/altitude (m)
    - 0.0              # z position (m)
    - 0.0              # x velocity (m/s)
    - 0.0              # y velocity (m/s)
    - 0.0              # z velocity (m/s)

control:
  mode: "hover"        # Control mode: hover, altitude_hold, manual
  target_altitude: 150.0  # meters
  max_thrust: 2000.0     # Newtons - Arc reactor max output
  kp: 2.0                # Proportional gain
  ki: 0.1                # Integral gain
  kd: 1.0                # Derivative gain

thrusters:
  main:
    location: [0, 0, 0]
    max_force: 1000.0
  left_hand:
    location: [-0.5, 0, 0]
    max_force: 500.0
  right_hand:
    location: [0.5, 0, 0]
    max_force: 500.0
  left_foot:
    location: [-0.3, -1, 0]
    max_force: 500.0
  right_foot:
    location: [0.3, -1, 0]
    max_force: 500.0
"""
    
    logger.info("Creating default configuration file...")
    config_path.write_text(config_content)
    logger.info(f"Configuration saved to: {config_path}")
    
    return str(config_path)

def run_basic_simulation():
    """Run the basic flight simulation."""
    logger.info("\n=== Starting Iron Man Suit Simulation ===\n")
    
    # Import simulation components
    try:
        from backend.aerodynamics.flight_models.flight_dynamics import FlightDynamics
        from backend.aerodynamics.environmental_effects.atmospheric_density import density_at_altitude
        
        logger.info("✓ Core modules loaded successfully")
    except ImportError as e:
        logger.error(f"Failed to import core modules: {e}")
        return False
    
    # Create configuration
    config_path = create_default_config()
    
    # Load and run simulation
    import yaml
    import numpy as np
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("\nInitializing flight dynamics...")
    suit_config = config["suit"]
    flight_dynamics = FlightDynamics(
        mass=suit_config["mass"],
        wing_area=suit_config["wing_area"],
        Cl0=suit_config["Cl0"],
        Cld_alpha=suit_config["Cld_alpha"],
        Cd0=suit_config["Cd0"],
        k=suit_config["k"]
    )
    
    # Run simulation
    sim_config = config["simulation"]
    control_config = config["control"]
    
    state = np.array(sim_config["initial_state"])
    dt = sim_config["dt"]
    duration = sim_config["duration"]
    
    logger.info(f"\nSimulation parameters:")
    logger.info(f"  Duration: {duration}s")
    logger.info(f"  Time step: {dt}s")
    logger.info(f"  Initial altitude: {state[1]}m")
    logger.info(f"  Target altitude: {control_config['target_altitude']}m")
    
    # Storage for results
    times = []
    states = []
    
    logger.info("\nRunning simulation...")
    t = 0.0
    last_log_time = 0.0
    
    while t <= duration:
        times.append(t)
        states.append(state.copy())
        
        # Simple altitude control
        altitude_error = control_config["target_altitude"] - state[1]
        thrust = control_config["kp"] * altitude_error
        thrust = np.clip(thrust, 0, control_config["max_thrust"])
        
        # Control input
        control = {"thrust": thrust, "alpha": 0.0}
        
        # Step dynamics
        state = flight_dynamics.step(state, control, dt)
        
        # Log progress every second
        if t - last_log_time >= 1.0:
            velocity = np.linalg.norm(state[3:6])
            logger.info(f"  t={t:5.1f}s | Alt: {state[1]:6.1f}m | Vel: {velocity:5.1f}m/s | Thrust: {thrust:6.1f}N")
            last_log_time = t
        
        t += dt
    
    logger.info("\n✓ Simulation completed successfully!")
    
    # Plot results
    try:
        import matplotlib.pyplot as plt
        
        times = np.array(times)
        states = np.array(states)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Altitude plot
        ax1.plot(times, states[:, 1], 'b-', label='Altitude')
        ax1.axhline(y=control_config["target_altitude"], color='r', linestyle='--', label='Target')
        ax1.set_ylabel('Altitude (m)')
        ax1.set_xlabel('Time (s)')
        ax1.grid(True)
        ax1.legend()
        ax1.set_title('Iron Man Suit - Altitude Control')
        
        # Velocity plot
        velocities = np.linalg.norm(states[:, 3:6], axis=1)
        ax2.plot(times, velocities, 'g-')
        ax2.set_ylabel('Velocity (m/s)')
        ax2.set_xlabel('Time (s)')
        ax2.grid(True)
        ax2.set_title('Total Velocity')
        
        plt.tight_layout()
        output_path = REPO_ROOT / "simulation_results.png"
        plt.savefig(output_path, dpi=150)
        logger.info(f"\n✓ Results saved to: {output_path}")
        
        plt.show()
        
    except ImportError:
        logger.warning("matplotlib not available - skipping visualization")
    
    return True

def main():
    """Main entry point."""
    logger.info("Iron Man Suit Simulation - Minimal Startup")
    logger.info("=" * 50)
    
    # Check requirements
    if not check_requirements():
        logger.error("Please install missing requirements and try again")
        return 1
    
    # Optional: Build physics plugin
    build_physics_plugin()
    
    # Run simulation
    if run_basic_simulation():
        logger.info("\n✓ All systems operational!")
        return 0
    else:
        logger.error("\n✗ Simulation failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())