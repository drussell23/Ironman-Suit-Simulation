# Flight Models

The `flight_models` package provides core aerodynamic and propulsion components for Iron Man suit simulation.

## Modules

### thrust_and_propulsion.py
- `compute_thrust(mass, acceleration, gravity=9.81, margin=1.0, min_thrust=0.0, max_thrust=None)`  
  Computes required thrust (N) to overcome gravity and achieve a target acceleration. Supports safety margin, thrust limits, input validation, and debug logging.

### aerodynamic_forces.py
- `aerodynamic_forces(velocity, alpha, altitude, wing_area, Cl0, Cld_alpha, Cd0, k, stall_angle, compressibility)`  
  Calculates lift and drag vectors using angle of attack, altitude-based density, stall modeling, optional Prandtl–Glauert compressibility correction, and detailed logging.

- Helper functions:
  - `compute_lift_coefficient` with stall saturation
  - `compute_drag_coefficient` with induced drag term

### flight_dynamics.py
- `FlightDynamics` class encapsulates 3-DOF motion:
  - **aerodynamic_coeffs**: lift & drag coefficients
  - **aerodynamic_forces**: world-frame lift & drag
  - **derivatives**: state derivatives including thrust and gravity
  - **step**: 4th-order RK integration of state

### stability_control.py
- `StabilityControl` class implements a PD controller for angular stabilization (roll, pitch, yaw). Computes control torques based on orientation and rate errors.

### sensor_models
See [sensor_models/README.md](sensor_models/README.md) for IMU and Pitot sensor simulations used in closed-loop control and state estimation.

## Usage Example
```python
import numpy as np
from backend.aerodynamics.flight_models import (
    compute_thrust, aerodynamic_forces,
    FlightDynamics, StabilityControl
)

# Propulsion
thrust = compute_thrust(mass=80, acceleration=2.0, margin=1.1)

# Aerodynamic forces
vel = np.array([10.0, 0.0, 0.0])
lift, drag = aerodynamic_forces(
    velocity=vel, alpha=0.1, altitude=1000,
    wing_area=0.5, Cl0=0.2, Cld_alpha=5.7, Cd0=0.03, k=0.1,
    stall_angle=np.deg2rad(15), compressibility=True
)

# Flight dynamics
fd = FlightDynamics(mass=80, wing_area=0.5, Cl0=0.2, Cld_alpha=5.7, Cd0=0.03, k=0.1)
state0 = np.zeros(6)  # [x,y,z,vx,vy,vz]
control = {"thrust": thrust, "alpha": 0.1}
state1 = fd.step(state0, control, dt=0.01)

# Stability control
sc = StabilityControl()
torques = sc.stabilize(
    current_orientation=np.zeros(3),
    current_rates=np.zeros(3)
)
```

## Logging
All modules emit `DEBUG`-level logs. To enable verbose output:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```
