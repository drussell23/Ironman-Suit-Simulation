# Iron Man Suit Simulation - Quick Start Guide

## Bare Minimum Setup

This guide will get you running the essential Iron Man suit flight simulation with minimal setup.

### Prerequisites

1. **Python 3.8+** installed
2. **Basic dependencies**:
   ```bash
   pip install numpy scipy pyyaml matplotlib
   ```

3. **Optional (for C++ physics acceleration)**:
   - CMake 3.10+
   - C++ compiler (GCC/Clang/MSVC)

### Quick Start

1. **Run the simulation**:
   ```bash
   python start_simulation.py
   ```

   This will:
   - Check for required packages
   - Build the physics plugin (if CMake available)
   - Create a default configuration
   - Run a 30-second hover simulation
   - Display results

### What's Running?

The bare minimum simulation includes:

1. **Flight Dynamics** (`backend/aerodynamics/flight_models/`)
   - Basic 6DOF physics
   - Aerodynamic forces (lift, drag)
   - Thrust control

2. **Environmental Effects** (`backend/aerodynamics/environmental_effects/`)
   - Atmospheric density model
   - Basic wind effects

3. **Simple Control System**
   - PID altitude controller
   - Hover stabilization

### Configuration

The simulation creates `simulation_config.yaml` with:
- Suit parameters (mass, aerodynamics)
- Initial conditions
- Control gains
- Thruster configuration

Edit this file to customize the simulation.

### Minimal Python Example

```python
# Minimal flight simulation
from backend.aerodynamics.flight_models.flight_dynamics import FlightDynamics
import numpy as np

# Create flight model
flight = FlightDynamics(mass=100.0, wing_area=2.0)

# Initial state: [x, y, z, vx, vy, vz]
state = np.array([0.0, 100.0, 0.0, 0.0, 0.0, 0.0])

# Simulate hover
for t in range(100):
    # Simple altitude hold
    thrust = 1000.0 * (150.0 - state[1]) / 50.0
    control = {"thrust": thrust, "alpha": 0.0}
    
    # Step physics
    state = flight.step(state, control, dt=0.01)
    print(f"t={t*0.01:.2f}s, altitude={state[1]:.1f}m")
```

### Advanced Features (Optional)

To enable additional features:

1. **C++ Physics Plugin**:
   ```bash
   cd backend/aerodynamics/physics_plugin
   mkdir build && cd build
   cmake ..
   make -j4
   ```

2. **AI/Adaptive Control**:
   ```bash
   pip install torch scikit-learn
   ```
   Then use `backend/adaptive_ai/` modules

3. **Web API**:
   ```bash
   pip install fastapi uvicorn
   cd backend/api
   uvicorn main:app --reload
   ```

4. **Unity Visualization**:
   - Open `simulation/unity_simulation_ui/Iron-Man-Suit-Simulation` in Unity
   - Press Play to connect to backend

### Troubleshooting

1. **ImportError**: Make sure you're in the repo root directory
2. **Physics plugin fails**: You can run without it (pure Python mode)
3. **No matplotlib**: Install with `pip install matplotlib` for plots

### Next Steps

- Explore `backend/aerodynamics/simulations/` for more examples
- Check `backend/adaptive_ai/` for AI-powered control
- See `backend/jarvis/` for voice control integration
- Look at `backend/computer_vision/` for sensor simulation

### System Architecture

```
Iron Man Simulation
├── Backend (Python/C++)
│   ├── aerodynamics/        # Core physics
│   ├── adaptive_ai/         # AI control
│   ├── jarvis/             # Voice assistant
│   └── api/                # REST endpoints
└── Simulation (Unity)
    └── unity_simulation_ui/ # 3D visualization
```

For full documentation, see the main README.md files in each module.