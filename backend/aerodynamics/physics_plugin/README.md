# Aerodynamics Physics Plugin

A high-performance C library for computational fluid dynamics (CFD) simulations with Python bindings. This plugin provides core aerodynamic calculations for the Iron Man suit simulation platform.

## Overview

The Aerodynamics Physics Plugin implements:
- **Unstructured mesh support** for complex geometries
- **k-� turbulence modeling** for realistic flow simulations
- **Actuator models** for control surfaces and thrust vectoring
- **Finite volume solver** with convection and diffusion
- **Python bindings** for seamless integration with the main simulation

## Architecture

```
physics_plugin/
   include/aerodynamics/    # C header files
      mesh.h              # Mesh data structures
      flow_state.h        # Flow field variables
      turbulence_model.h  # k-� turbulence model
      actuator.h          # Force/momentum actuators
      solver.h            # CFD solver
      bindings.h          # C bindings for Python
   src/aerodynamics/       # C implementation
      mesh.c              # Mesh operations and convection
      flow_state.c        # Flow state management
      turbulence_model.c  # Turbulence calculations
      actuator.c          # Actuator forces
      solver.c            # Main solver loop
      bindings.c          # Python interface
   python/                 # Python bindings
      bindings.py         # Basic bindings
      bindings_improved.py # Enhanced bindings with error handling
   tests/                  # Unit tests
      aerodynamics/       # C tests using CMocka
   cmake/                  # Build configuration

```

## Building

### Prerequisites
- CMake 3.10+
- C compiler (GCC, Clang, or MSVC)
- Python 3.8+ (for bindings)
- CMocka (for testing)

### Build Instructions

```bash
# From the physics_plugin directory
mkdir build && cd build
cmake -DENABLE_TESTING=ON ..
cmake --build . -- -j4
```

### Platform-Specific Notes

**macOS:**
- Library will be built as `libaerodynamics_physics_plugin.dylib`
- Ensure Xcode command line tools are installed

**Linux:**
- Library will be built as `libaerodynamics_physics_plugin.so`
- May need to set `LD_LIBRARY_PATH` for runtime loading

**Windows:**
- Library will be built as `aerodynamics_physics_plugin.dll`
- Use Visual Studio generator: `cmake -G "Visual Studio 16 2019" ..`

## Python Integration

### Basic Usage

```python
from aerodynamics.physics_plugin.python import bindings

# Create a simple tetrahedral mesh
mesh = bindings.Mesh.tetrahedron(size=1.0)

# Set up turbulence model (k-epsilon)
turb_model = bindings.TurbulenceModel(
    c_mu=0.09,
    sigma_k=1.0,
    sigma_eps=1.3,
    c1_eps=1.44,
    c2_eps=1.92
)

# Create solver
solver = bindings.Solver(mesh, turb_model)
solver.initialize()

# Create actuator for thrust control
actuator = bindings.Actuator(
    name="thruster",
    node_ids=[0, 1],
    direction=[0.0, 0.0, 1.0],
    gain=1000.0,
    type=bindings.ACTUATOR_TYPE_BODY_FORCE
)
actuator.set_command(0.5)  # 50% thrust

# Run simulation
dt = 0.001  # 1ms timestep
for step in range(100):
    solver.apply_actuator(actuator, dt)
    solver.step(dt)

# Extract results
flow_state = bindings.FlowState(mesh)
solver.read_state(flow_state)
velocity = flow_state.get_velocity()
pressure = flow_state.get_pressure()
```

### Enhanced Bindings with Error Handling

The `bindings_improved.py` module provides robust error handling and cross-platform support:

```python
from aerodynamics.physics_plugin.python import bindings_improved as aero

try:
    # Create mesh with validation
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    cells = np.array([[0, 1, 2, 3]])
    mesh = aero.Mesh(vertices, cells)
    
    # Solver with automatic resource management
    solver = aero.Solver(mesh, aero.TurbulenceModel())
    solver.initialize()
    
except aero.AerodynamicsError as e:
    print(f"Aerodynamics error: {e}")
```

## Core Components

### Mesh
- Supports tetrahedral and hexahedral elements
- Automatic computation of cell volumes and adjacency
- Least-squares gradient reconstruction for unstructured grids

### Flow State
- Stores velocity, pressure, and turbulence quantities at nodes
- Efficient memory layout for cache performance
- Thread-safe read operations

### Turbulence Model
- Standard k-� model with wall functions
- Computes eddy viscosity for momentum diffusion
- Configurable model constants

### Actuators
- Three types: SURFACE, BODY_FORCE, VOLUME
- Apply forces at specified mesh nodes
- Used for modeling thrusters, control surfaces, etc.

### Solver
- Explicit time-stepping scheme
- Operator splitting: convection � diffusion � turbulence
- Modular design for easy extension

## Testing

Run the test suite:

```bash
cd build
ctest --output-on-failure
```

Individual test executables:
- `test_mesh` - Mesh operations
- `test_flow_state` - Flow field management
- `test_turbulence_model` - Turbulence calculations
- `test_actuator` - Force application
- `test_solver` - Integration tests
- `test_bindings` - Python interface

## Performance Considerations

- **Memory Layout**: Structures are cache-aligned for optimal performance
- **Vectorization**: Inner loops are written to enable compiler auto-vectorization
- **Minimal Allocations**: Solver reuses memory buffers across timesteps
- **Future Optimizations**:
  - OpenMP parallelization (planned)
  - GPU acceleration via CUDA/OpenCL (planned)
  - SIMD intrinsics for critical loops

## Known Limitations

1. Currently only supports tetrahedral meshes (hexahedral support partial)
2. Single-threaded execution (OpenMP support planned)
3. No adaptive mesh refinement
4. Limited to incompressible flows
5. No implicit time-stepping schemes

## Future Enhancements

### Near-term (Q1 2025)
-  Enhanced error handling in Python bindings
- � OpenMP parallelization for multi-core CPUs
- � VTK output for ParaView visualization

### Medium-term (Q2-Q3 2025)
- � GPU acceleration (CUDA/OpenCL)
- � Large Eddy Simulation (LES) turbulence model
- � Detached Eddy Simulation (DES) hybrid model
- � Adaptive mesh refinement (AMR)

### Long-term
- Multigrid solvers for faster convergence
- Fluid-structure interaction (FSI)
- Compressible flow support
- MPI parallelization for clusters

## API Reference

### C API

See header files in `include/aerodynamics/` for detailed function documentation.

Key functions:
- `mesh_create()` - Create mesh from vertices and connectivity
- `solver_create()` - Initialize solver with mesh and turbulence model
- `solver_step()` - Advance solution by one timestep
- `actuator_apply()` - Apply actuator forces to flow field

### Python API

See docstrings in `python/bindings_improved.py` for comprehensive documentation.

Key classes:
- `Mesh` - Computational mesh
- `TurbulenceModel` - k-� turbulence model
- `Actuator` - Force/momentum source
- `Solver` - CFD solver
- `FlowState` - Flow field container

## Contributing

When contributing to the physics plugin:

1. **Code Style**: Follow the existing C style (K&R with 4-space indents)
2. **Testing**: Add unit tests for new functionality
3. **Documentation**: Update this README and add docstrings
4. **Performance**: Profile changes to ensure no regressions
5. **Portability**: Test on macOS, Linux, and Windows

## License

This module is part of the Iron Man Suit Simulation project. See the main project LICENSE file for details.

## Contact

For questions or issues specific to the physics plugin, please open an issue in the main Iron Man repository with the `[physics-plugin]` tag.