# Aerodynamics Physics Plugin (C)

**Version**: 1.0.0  
A standalone C library exposing a core CFD‐solver plugin with k–ε turbulence, LES subgrid models, actuators, mesh utilities, and a programmable API—complete with a full suite of CMocka unit tests.

---

## Prerequisites

- **CMake** ≥ 3.18  
- **C compiler** supporting C11 (Clang, GCC)  
- Internet access (FetchContent will pull in CMocka)  

---

## Building & Running C Tests

1. **Configure & generate build files (enable unit tests)**

```bash
mkdir -p build
cd build
cmake .. -DENABLE_TESTING=ON
```

2. Compile the library and tests

```bash
cmake --build .
```

3. (Optional) Run all tests via CTest

```bash
ctest --output-on-failure
```

---

## Directory Layout

```text
physics_plugin/
├── CMakeLists.txt
├── cmake/
│   ├── aerodynamics_physics_plugin_config.h.in
│   ├── Config.cmake.in
│   └── aerodynamics_physics_plugin.pc.in
├── include/aerodynamics/
│   ├── actuator.h
│   ├── flow_state.h
│   ├── mesh.h
│   ├── solver.h
│   └── turbulence_model.h
├── src/aerodynamics/
│   ├── actuator.c
│   ├── bindings.c
│   ├── flow_state.c
│   ├── mesh.c
│   ├── solver.c
│   └── turbulence_model.c
├── tests/aerodynamics/
│   ├── test_actuator.c
│   ├── test_bindings.c
│   ├── test_flow_state.c
│   ├── test_mesh.c
│   ├── test_solver.c
│   └── test_turbulence_model.c
├── scripts/
│   └── fix_eof.sh
└── README.md        ← *(this file)*
```