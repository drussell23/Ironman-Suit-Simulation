# Aerodynamics Plugin Unit Tests

This directory houses **CMocka**-based unit tests for the individual modules of the Aerodynamics physics plugin. 

---

## Structure

```bash
tests/aerodynamics/
├── test_actuator.c             # Tests for `actuator.c`
├── test_bindings.c             # Tests for the C binding layer
├── test_flow_state.c           # Tests for `flow_state.c`
├── test_mesh.c                 # Tests for `mesh.c`
├── test_solver.c               # Tests for `solver.c`
└── test_turbulence_model.c     # Tests for `turbulence_model.c` 
```

---

## Running the tests

1. **Configure and build** (from `physics_plugin` root):

```bash
mkdir -p build && cd build
cmake -DENABLE_TESTING=ON ..
make -j$(nproc)
```

2. **Run all unit tests:**

```bash
ctest --output-on-failure -R test_
```

3. **Run a single test** (e.g., mesh tests):

```bash
ctest -R test_mesh --output-on-failure
```

---

## Adding new tests

1. Create a new `test_*.c` file following the naming convention.
2. Write CMocka `static void` test functions and register them in your file's `main()`.

3. Re-run CMake to pick up the new file automatically. 

---

## Dependencies

* **CMake** ≥ 3.10
* cmocka pulled in via `FetchContent` in `CMakeLists.txt`