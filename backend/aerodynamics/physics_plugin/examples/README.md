# Full Pipeline Example

This example demonstrates a complete aerodynamics simulation pipeline using the Aerodynamics Physics Plugin C API. It covers:

- Mesh creation (tetrahedral mesh)
- FlowState initialization (velocity & pressure)
- Turbulence modeling (k–ε)
- Actuator creation & application (surface & body-force)
- Solver setup and time stepping
- Diagnostics (velocity, pressure, runtime)
- Cleanup of all objects

## Directory Structure

```plain
examples/
├── CMakeLists.txt           Build script for this example
├── full_pipeline_example.c  Source code of the demo
└── build/                   CMake build directory (created at build time)
```

## Prerequisites

- CMake ≥ 3.10
- C compiler with C99 support (AppleClang, GCC, etc.)
- The Aerodynamics Physics Plugin built (`libaerodynamics_physics_plugin.dylib` or `.so`)
- On macOS, ensure matching architectures (arm64 vs x86_64) or build a universal binary.

## Building the Example

1. **Build the plugin (if not already):**
   ```bash
   cd backend/aerodynamics/physics_plugin
   mkdir -p build && cd build
   cmake .. \
     -DCMAKE_OSX_ARCHITECTURES=arm64         # or x86_64, or "arm64;x86_64"
   cmake --build .
   ```

2. **Build the example:**
   ```bash
   cd ../examples
   mkdir -p build && cd build
   cmake .. \
     -DCMAKE_OSX_ARCHITECTURES=arm64         # must match plugin build
     [-DCMAKE_BUILD_TYPE=Release]            # optional: Debug or Release
   cmake --build .
   ```

> **Note**: You can adjust `CMAKE_OSX_ARCHITECTURES` for your target or embed it in `CMakeLists.txt`:
> ```cmake
> if(APPLE AND NOT DEFINED CMAKE_OSX_ARCHITECTURES)
>   set(CMAKE_OSX_ARCHITECTURES arm64 CACHE STRING "" FORCE)
> endif()
> ```

## Running the Demo

From `examples/build`:

```bash
./full_pipeline_example [dt] [steps]
```

- `dt`    : timestep size (default `0.1`)
- `steps` : number of timesteps (default `10`)

**Examples:**
```bash
./full_pipeline_example            # dt=0.1, steps=10
./full_pipeline_example 0.05 20    # dt=0.05, steps=20
```

## Expected Output

You should see:

```
Extended Aerodynamics Pipeline Example
 Time step dt = 0.1000, steps = 10

Mesh: 4 nodes, 1 cells
 Cell 0: nodes [0,1,2,3], volume = 0.166667
 Mesh adjacency offsets: 0 0

Initialized flow state (vel & press zeroed)

Turbulence model initialized (k-epsilon)

Initial eddy viscosity at nodes: 0.0009 0.0009 0.0009 0.0009

Created actuators: wing(surface), thruster(body-force)

Solver initialized

 Step 0: cmd(wing=0.00 thr=0.00), avg_vel=0.000, avg_pres=0.000
 ...
 Step 9: cmd(wing=0.90 thr=1.80), avg_vel=3.230, avg_pres=0.000

Completed 10 steps in 0 seconds

Final velocities & pressures:
 Node 0: vel=(-1.6792,4.3483,0.0000), p=0.0000
 Node 1: vel=(-1.6792,4.3483,0.0000), p=0.0000
 Node 2: vel=(1.7994,0.0000,0.0000), p=0.0000
 Node 3: vel=(1.7994,0.0000,0.0000), p=0.0000
```

## Troubleshooting

- **`dyld: library not loaded`** on macOS: ensure RPATH is set correctly in `CMakeLists.txt`:
  ```cmake
  if(APPLE)
    set_target_properties(full_pipeline_example PROPERTIES
      INSTALL_RPATH "@loader_path/../../build"
      BUILD_WITH_INSTALL_RPATH TRUE
    )
  endif()
  ```
- **Segmentation faults**: rebuild with `-DCMAKE_BUILD_TYPE=Debug`, run under `lldb`, and use `bt` to inspect.
- **Typedef warnings**: harmless redefinition between headers; can be silenced with compiler flags.


## Extending the Example

- Modify `full_pipeline_example.c` to:
  - Use a more complex mesh.
  - Add new actuators or actuator groups.
  - Query intermediate fields (k, ε, eddy viscosity).
  - Integrate boundary conditions.
- Incorporate this demo into a larger CMake superproject by enabling examples in the main plugin CMake:
  ```bash
  cmake -DENABLE_EXAMPLES=ON ..
  ```

---
Built and tested on macOS (arm64). Adjust flags for other platforms.
