# Aerodynamics Physics Plugin

**Version**: 1.0.0  
**Description**:  
Shared C library providing a CFD solver with k–ε turbulence models, Smagorinsky LES, actuator support, and Python bindings.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)  
2. [Directory Layout](#directory-layout)  
3. [Building](#building)  
4. [Running Tests](#running-tests)  
5. [Python Usage](#python-usage)  
6. [pkg-config & CMake Integration](#pkg-config--cmake-integration)  
7. [Cleaning Up](#cleaning-up)  

---

## Prerequisites

- **CMake** ≥ 3.18  
- **C compiler**: Clang/GCC (supports C11)  
- **CMocka** (auto-fetched via FetchContent)  
- **Python** ≥ 3.7 (for Python bindings)  
- **NumPy** (for Python examples)

---

## Directory Layout

```text
physics_plugin/
├── CMakeLists.txt
├── cmake/
│   ├── aerodynamics_physics_plugin_config.h.in
│   ├── Config.cmake.in
│   └── aerodynamics_physics_plugin.pc.in
├── include/
│   └── aerodynamics/
│       ├── solver.h
│       ├── mesh.h
│       ├── flow_state.h
│       ├── actuator.h
│       └── turbulence_model.h
├── src/
│   ├── solver.c
│   ├── mesh.c
│   ├── flow_state.c
│   ├── actuator.c
│   └── turbulence_model.c
├── tests/
│   ├── test_solver.c
│   ├── test_mesh.c
│   ├── test_flow_state.c
│   ├── test_actuator.c
│   └── test_turbulence_model.c
├── python/
│   └── command_executor.py
└── README.md         ← *(this file)*
