# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Iron Man Suit Simulation is an advanced software platform simulating a powered exoskeleton with technologies including:
- Flight dynamics and aerodynamics (CFD solver, turbulence models)
- AI systems (reinforcement learning, tactical decision making)
- Hardware integration (embedded OS, drivers, real-time control)
- Computer vision and sensor fusion
- JARVIS-style voice assistant
- Unity-based 3D visualization

## Key Development Commands

### Environment Setup
```bash
# Initial setup (creates venv, installs deps, builds C++ components)
./scripts/setup_env.sh

# Activate virtual environment
source venv/bin/activate
```

### Building C++ Components
```bash
cd backend/aerodynamics/physics_plugin
rm -rf build && mkdir build && cd build
cmake -DENABLE_TESTING=ON ..
cmake --build . -- -j4
```

### Running Tests

**Python Tests:**
```bash
# From project root
cd backend
python -m pytest aerodynamics/tests/ -v

# Run specific test file
python -m pytest aerodynamics/tests/flight_models/test_flight_dynamics.py -v

# With coverage
python -m pytest aerodynamics/tests/ -v --cov=aerodynamics
```

**C++ Tests:**
```bash
cd backend/aerodynamics/physics_plugin/build
ctest --output-on-failure

# Run specific test
./test_actuator
```

### Running Simulations
```bash
# Aerodynamics simulation
python backend/aerodynamics/simulations/run_simulation.py --plot

# With custom config
python backend/aerodynamics/simulations/run_simulation.py --config backend/aerodynamics/simulations/configs/suit_hover.yaml
```

## High-Level Architecture

### Backend Structure
- **adaptive_ai/**: Advanced AI systems including reinforcement learning (DQN, PPO, SAC), tactical decision making, behavioral adaptation, and cognitive load management
- **aerodynamics/**: Core flight physics with both Python and C implementations
  - **physics_plugin/**: High-performance C library for CFD, turbulence models, and aerodynamic calculations
  - **flight_models/**: Python flight dynamics, stability control, thrust systems
  - **control/**: Autopilot, guidance systems, actuator control
- **api/**: FastAPI-based REST endpoints for all subsystems
- **jarvis/**: Voice assistant with NLP, intent recognition, and command execution
- **computer_vision/**: Object detection, feature tracking, camera models
- **quantum_computing/**: Quantum algorithms and optimization (experimental)
- **systems_programming/**: Low-level embedded OS, drivers, real-time control

### Key Design Patterns
1. **Plugin Architecture**: C libraries (like aerodynamics_physics_plugin) expose functionality via clean APIs with Python bindings
2. **Modular AI Systems**: Each AI component (RL, tactical, behavioral) can operate independently or be integrated via ai_system.py
3. **Hardware Abstraction**: systems_programming/ provides HAL for portability across embedded platforms
4. **Event-Driven Control**: Real-time systems use event-driven architecture for responsive control

### Critical Integration Points
- Python/C boundaries: aerodynamics/physics_plugin/python/bindings.py wraps C functions
- AI/Control integration: adaptive_ai systems output commands to aerodynamics/control modules
- Sensor fusion: Multiple sensor models (IMU, pitot, lidar, radar) feed into state estimation
- Unity visualization: Unity project in simulation/unity_simulation_ui/ connects via network protocols

## Development Workflow

1. For aerodynamics work: Focus on backend/aerodynamics/, rebuild C++ plugin after changes
2. For AI development: Work in backend/adaptive_ai/, use reinforcement learning environments
3. For embedded systems: systems_programming/ contains drivers and RTOS components
4. Always run relevant tests after changes (pytest for Python, ctest for C++)
5. The project uses scientific Python stack (numpy, scipy, torch) extensively

## Important Notes
- Modified files show in git status (reinforcement_learning.py, tactical_decision.py)
- Many __pycache__ and build artifacts were recently deleted
- Project supports macOS, Linux, and Windows (with platform-specific code)
- Unity simulation requires separate Unity Editor setup