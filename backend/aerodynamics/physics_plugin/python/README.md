# Python Bindings & Tests for Aerodynamics Plugin 

This folder contains:

- **`bindings.py`**
  A set of `ctypes`-based Python wrappers around the core Aerodynamics C library (`libaerodynamics_physics_plugin.dylib`). It exposes high-level classes (`Mesh`, `FlowState`, `TurbulenceModel`, `Solver`, `Actuator`, etc.) that can be used from Python. 

- **`command_executor.py`** and **`test_command_executor.py`**
  A small utility (`command_executor.py`) that runs shell commands and parse JSON output. Its associated unit test file (`test_command_executor.py`) lives here as well.

- **`integration_tests/test_full_pipeline.py`**
  An end-to-end smoke test that:
  1. Builds a simple mesh, 
  2. Creates a turbulence model, 
  3. Constructs an actuator, 
  4. Initializes the solver, 
  5. Steps through a few timestamps, and 
  6. Verifies that no numerical `NaN` values appear in the final velocity/pressure arrays.

## Prerequisites 

1. **C Shared Library (`libaerodynamics_physics_plugin.dylib`)**
   Make sure you have already built the Aerodynamics C plugin (with all its unit tests passing) so that:

   **`physics_plugin/build/libaerodynamics_physics_plugin.dylib`** exists. On Apple Silicon, you must compile for `arm64` (or create a universal binary). For example:

```bash
cd path/to/IronMan/backend/aerodynamics/physics_plugin
rm -rf build 
mkdir build && cd build
cmake -DCMAKE_OSX_ARCHITECTURES=arm64 ..
make
```

After this, **`build/libaerodynamics_physics_plugin.dylib`** should be an **`arm64`** Mach-O library. You can verify with:

file **`build/libaerodynamics_physics_plugin.dylib`** 
# => Mach-O 64-bit dynamically linked shared library arm64

2. **Python & Dependencies**
    * Python 3.8+ (tested on Python 3.10).
    * **`pytest`** or **`unittest`** (built-in) for running tests.
    * **`numpy`** (used by **`FlowState.get_velocity()`** and **`get_pressure()`**).

   If you don't already have **`numpy`**, install it via pip: **`pip install numpy`**.

## Directory Layout 

```bash
python/
├── __pycache__/                 # (automatically generated)
├── bindings.py                  # Python–C bindings for Aerodynamics plugin
├── command_executor.py          # Helper for running external commands (shell) from Python
├── test_command_executor.py     # Unit test for command_executor.py
├── README.md                    # (this file)
└── integration_tests/
    ├── __pycache__/             # (auto)
    └── test_full_pipeline.py    # End-to-end integration test for bindings.py
```

## How to Run Unit Tests

1. Test **`command_executor.py`** (example of a simple unit test)

```bash 
cd path/to/IronMan/backend/aerodynamics/physics_plugin/python
python -m unittest test_command_executor.py
# or, if you prefer pytest:
pytest test_command_executor.py
```

This verifies that your **`run_and_parse_json()`** and associated helpers in **`command_executor.py`** behave as expected.

## How to Run the Integration Test (`test_full_pipeline.py`)

Because **`bindings.py`** expects to find **`libaerodynamics_physics_plugin.dylib`** in **`../build/`**, you must ensure:

1. You have built the C plugin as described above.
2. Python's import path includes the **`python/`** folder itself so that **`bindings.py`** can be imported by the test.

Below are two common ways to run the integration test: 

## Option A: From within `integration_tests/` with `PYTHONPATH`

1. Open a terminal and navigate into the integration test folder:

**`cd path/to/IronMan/backend/aerodynamics/physics_plugin/python/integration_tests`**

2. Prepend the parent directory (**`..`**) onto **`PYTHONPATH`** so Python can locate **`bindings.py`**: 

**`export PYTHONPATH="../:$PYTHONPATH"`**

3. Run the test directly:

**`python test_full_pipeline.py`**

or using **`unittest`** discovery:

**`python -m unittest test_full_pipeline.py`**

If everything is set up correctly, you will see output similar to:

```csharp 
[AeroBindings] Mesh created successfully (nodes: 4, cells: 1)
[AeroBindings] Turbulence model created (Cmu=0.090, sigma_k=1.000)
[AeroBindings] Actuator 'A0' created (type: 0, nodes: 1)
[AeroBindings] Solver created successfully.
[AeroBindings] Solver initialized.
[AeroBindings] Flow state created.
[AeroBindings] Actuator applied to solver for dt=0.05000 seconds.
[AeroBindings] Solver stepped forward by dt=0.05000 seconds.
[AeroBindings] Solver’s flow state copied into FlowState object.
… (repeats for each timestep) …
Final Velocities: [0. 0. 0. 0. 0. …]
Final Pressures:  [0. 0. 0. 0.]
[AeroBindings] Actuator destroyed.
[AeroBindings] Solver destroyed.
[AeroBindings] Turbulence model destroyed.
[AeroBindings] Flow state destroyed.
[AeroBindings] Mesh destroyed.

----------------------------------------------------------------------
Ran 1 test in 0.004s

OK
```

That **`OK`** at the bottom confirms a successful integration smoke test.

## Option B: From the python/ root folder

If you'd rather avoid exporting **`PYTHONPATH`** manually, you can run the integration test from the parent directory (**`python/`**) where **`bindings.py`** lives:

```bash
cd path/to/IronMan/backend/aerodynamics/physics_plugin/python
python -m unittest integration_tests.test_full_pipeline
```

Because **`python/`** is on **`sys.path[0]`**, Python can find **`bindings.py`** automatically. You should see the same successful output as above.

## Troubleshooting

1. **`ModuleNotFoundError: No module named 'bindings'`**
* Make sure you have `export PYTHONPATH="../"` if you are inside `integration_tests/`.
* Or run from the `python/` folder so that `bindings.py` is on the default import path.

2. **`OSError: dlopen(.../build/libaerodynamics_physics_plugin.dylib, ...) incompatible architecture`**
* That means your `.dylib` is built for the wrong CPU (e.g. x86_64 vs. arm64).
* Rebuild the library for your machine’s architecture. On Apple Silicon (M1/M2), run:

```bash
cd path/to/physics_plugin/build
rm -rf *
cmake -DCMAKE_OSX_ARCHITECTURES=arm64 ..
make
```

* Confirm with:

```bash 
file build/libaerodynamics_physics_plugin.dylib
# Should say “Mach-O 64-bit dynamically linked shared library arm64”
```

3. **`TypeError` from `solver.step()` or `apply_actuator()`**
* Ensure your `bindings.py` wrapper for `solver_step_bind` and `solver_apply_actuator_bind` passes `ctypes.c_double(dt)` (not a plain Python float).
* Also verify that you declared:

```python
_lib.solver_step_bind.argtypes = [ctypes.c_void_p, ctypes.c_double]
_lib.solver_apply_actuator_bind.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_double]
```

4. **`actuator_set_command_bind` missing**
* If your C code defines `actuator_set_command_bind(...)`, be sure you added the corresponding:

```python
_lib.actuator_set_command_bind.argtypes = [ctypes.c_void_p, ctypes.c_double]
_lib.actuator_set_command_bind.restype  = None

class Actuator:
    …
    def set_command(self, cmd: float):
        _lib.actuator_set_command_bind(self._as_parameter_, ctypes.c_double(cmd))
```
* Otherwise, your Python test may fail if it tries to call `set_command()`.







