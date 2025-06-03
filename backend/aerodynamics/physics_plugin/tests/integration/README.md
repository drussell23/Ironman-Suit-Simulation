# Integration Smoke Test

This directory contains the **end-to-end integration** (smoke) test for the Aerodynamics physics plugin, which verifies that all modules wire together correctly. 

---

## File

```bash
tests/integration/
└── test_full_pipeline.c      # Smoke test: mesh → flow_state → actuator → turbulence → solver
```

---

## Purpose

* **Mesh topology:** Constructs a 4-node tetrahedral mesh via the full mesh_create(...) API.

* **Flow state:** Initializes the flow variables (velocity, pressure, turbulence) on the mesh.

* **Solver:** Creates and initializes the solver, applies an actuator, runs multiple timesteps. 

* **Turbulence model:** Updates only the k–ε fields to ensure no uninitialized memory is touched. 

* **Validation:** Reads back the final flow state and asserts there are **no NaNs** in velocity or pressure. 

---

## Running the smoke test

1. **From the `physics_plugin` root:**

```bash 
mkdir -p build && cd build
cmake -DENABLE_TESTING=ON ..
make -j$(nproc)
```

2. **Invoke only the full-pipeline test:**

```bash
ctest -R full_pipeline_c --output-on-failure
```

3. **To run all tests (unit + integration):**

```bash
ctest --output-on-failure
```

---

## Extending the smoke test

* **Geometry:** Swap out the single-tetra mesh for a larger mesh.

* **Actuator scenarios:** Add more actuators or time-dependent commands. 

* **Turbulence physics:** Re-enable full velocity-gradient updates when ready.

Once this smoke test passes, you have confidence that the core C pipeline operates end-to-end. Feel free to extend it to more complex cases!