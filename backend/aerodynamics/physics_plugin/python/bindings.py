import os
import ctypes
import numpy as np
from ctypes.util import find_library

# -----------------------------------------------------------------------------
# Load the shared library (more robust than a fixed Path())
# -----------------------------------------------------------------------------
_lib_name = "aerodynamics_physics_plugin"
# Try a relative path first
_lib_path = os.path.join(
    os.path.dirname(__file__), "..", "build", f"lib{_lib_name}.dylib"
)
if not os.path.exists(_lib_path):
    # Try .so for Linux
    _lib_path = os.path.join(
        os.path.dirname(__file__), "..", "build", f"lib{_lib_name}.so"
    )
    if not os.path.exists(_lib_path):
        # Fall back to system search (e.g. /usr/local/lib)
        _found = find_library(_lib_name)
        if not _found:
            raise OSError(f"Could not find shared library '{_lib_name}'")
        _lib_path = _found

_lib = ctypes.CDLL(_lib_path)

# -----------------------------------------------------------------------------
# Error handling decorators
# -----------------------------------------------------------------------------
class AerodynamicsError(Exception):
    """Exception raised for errors in the aerodynamics physics plugin."""
    pass

def check_null_result(func):
    """Decorator to check if C function returns NULL and raise exception."""
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if result is None or (isinstance(result, ctypes.c_void_p) and not result):
            raise AerodynamicsError(f"{func.__name__} returned NULL")
        return result
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper

# ----------------------------------------------------------------------
# Python bindings for Aerodynamics plugin continue below...
# ----------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Handle types
# -----------------------------------------------------------------------------
MeshHandle = ctypes.c_void_p
TurbulenceModelHandle = ctypes.c_void_p
ActuatorHandle = ctypes.c_void_p
SolverHandle = ctypes.c_void_p
FlowStateHandle = ctypes.c_void_p

# -----------------------------------------------------------------------------
# C function prototypes
# -----------------------------------------------------------------------------
_lib.mesh_create_bind.restype = MeshHandle
_lib.mesh_create_bind.argtypes = [
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t),
]
_lib.mesh_destroy_bind.restype = None
_lib.mesh_destroy_bind.argtypes = [MeshHandle]
_lib.mesh_get_num_nodes.restype = ctypes.c_size_t
_lib.mesh_get_num_nodes.argtypes = [MeshHandle]

_lib.turb_model_create_bind.restype = TurbulenceModelHandle
_lib.turb_model_create_bind.argtypes = [
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
]
_lib.turb_model_destroy_bind.restype = None
_lib.turb_model_destroy_bind.argtypes = [TurbulenceModelHandle]

_lib.actuator_create_bind.restype = ActuatorHandle
_lib.actuator_create_bind.argtypes = [
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_double,
]
_lib.actuator_set_command_bind.restype = None
_lib.actuator_set_command_bind.argtypes = [ActuatorHandle, ctypes.c_double]
_lib.actuator_destroy_bind.restype = None
_lib.actuator_destroy_bind.argtypes = [ActuatorHandle]

_lib.solver_create_bind.restype = SolverHandle
_lib.solver_create_bind.argtypes = [MeshHandle, TurbulenceModelHandle]
_lib.solver_initialize_bind.restype = None
_lib.solver_initialize_bind.argtypes = [SolverHandle]
_lib.solver_step_bind.restype = None
_lib.solver_step_bind.argtypes = [SolverHandle, ctypes.c_double]
_lib.solver_apply_actuator_bind.restype = None
_lib.solver_apply_actuator_bind.argtypes = [
    SolverHandle,
    ActuatorHandle,
    ctypes.c_double,
]
_lib.solver_destroy_bind.restype = None
_lib.solver_destroy_bind.argtypes = [SolverHandle]

_lib.flow_state_create_bind.restype = FlowStateHandle
_lib.flow_state_create_bind.argtypes = [MeshHandle]
_lib.solver_read_state_bind.restype = None
_lib.solver_read_state_bind.argtypes = [SolverHandle, FlowStateHandle]
_lib.flow_state_destroy_bind.restype = None
_lib.flow_state_destroy_bind.argtypes = [FlowStateHandle]
_lib.flow_state_get_velocity_bind.restype = None
_lib.flow_state_get_velocity_bind.argtypes = [
    FlowStateHandle,
    ctypes.POINTER(ctypes.c_double),
]
_lib.flow_state_get_pressure_bind.restype = None
_lib.flow_state_get_pressure_bind.argtypes = [
    FlowStateHandle,
    ctypes.POINTER(ctypes.c_double),
]


# -----------------------------------------------------------------------------
# Python wrapper classes
# -----------------------------------------------------------------------------
class Mesh:
    """Represent a tetrahedral mesh."""

    def __init__(self, coords, connectivity):
        n = len(coords) // 3
        coords_arr = (ctypes.c_double * (3 * n))(*coords)
        c = len(connectivity)
        conn_arr = (ctypes.c_size_t * c)(*connectivity)
        handle = _lib.mesh_create_bind(n, coords_arr, 1, c, conn_arr)

        if not handle:
            raise MemoryError("Failed to create Mesh.")

        self._as_parameter_ = handle

    @classmethod
    def tetrahedron(cls, size=1.0):
        """Create a unit tetrahedron mesh."""
        coords = [0.0, 0.0, 0.0, size, 0.0, 0.0, 0.0, size, 0.0, 0.0, 0.0, size]
        connectivity = [0, 1, 2, 3]

        return cls(coords, connectivity)

    def __del__(self):
        if getattr(self, "_as_parameter_", None):
            _lib.mesh_destroy_bind(self._as_parameter_)

    @property
    def num_nodes(self):
        return _lib.mesh_get_num_nodes(self._as_parameter_)


class TurbulenceModel:
    """Wraps k–ε turbulence model."""

    def __init__(self, c_mu=0.09, sigma_k=1.0, sigma_eps=1.3, c1_eps=1.44, c2_eps=1.92):
        handle = _lib.turb_model_create_bind(c_mu, sigma_k, sigma_eps, c1_eps, c2_eps)

        if not handle:
            raise MemoryError("Failed to create TurbulenceModel")

        self._as_parameter_ = handle

    def __del__(self):
        if getattr(self, "_as_parameter_", None):
            _lib.turb_model_destroy_bind(self._as_parameter_)


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Updated Actuator class
# ----------------------------------------------------------------------
# First, declare the argtypes and restype for the C function
_lib.actuator_create_bind.argtypes = [
    ctypes.c_char_p,  # Name
    ctypes.c_int,  # Type
    ctypes.c_int,  # n (number of node IDs)
    ctypes.POINTER(ctypes.c_size_t),  # node_ids array ()
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_double,
]
_lib.actuator_create_bind.restype = ctypes.c_void_p


class Actuator:
    """Wraps an actuator (e.g. surface, volumetric)."""

    def __init__(
        self,
        name: str,
        node_ids: list[int],
        direction: list[float],
        gain: float = 1.0,
        type: int = 0,
    ):
        name_b = name.encode("utf-8")
        type_c = ctypes.c_int(type)
        n = len(node_ids)
        nodes_arr = (ctypes.c_int * n)(*node_ids)
        dir_arr = (ctypes.c_double * 3)(*direction)
        gain_c = ctypes.c_double(gain)

        handle = _lib.actuator_create_bind(
            name_b,
            type_c,
            ctypes.c_int(n),
            ctypes.cast(nodes_arr, ctypes.POINTER(ctypes.c_size_t)),
            ctypes.cast(dir_arr, ctypes.POINTER(ctypes.c_double)),
            gain_c,
        )

        if not handle:
            raise RuntimeError("Failed to create actuator")
        self._as_parameter_ = handle

    def set_command(self, cmd):
        _lib.actuator_set_command_bind(self._as_parameter_, cmd)

    def __del__(self):
        if getattr(self, "_as_parameter_", None):
            _lib.actuator_destroy_bind(self._as_parameter_)


class FlowState:
    """Holds velocity and pressure arrays for a mesh."""

    def __init__(self, mesh: Mesh):
        handle = _lib.flow_state_create_bind(mesh._as_parameter_)

        if not handle:
            raise MemoryError("Failed to create FlowState.")

        self._as_parameter_ = handle
        self._mesh = mesh

    def get_velocity(self):
        n = self._mesh.num_nodes
        arr = (ctypes.c_double * (3 * n))()
        _lib.flow_state_get_velocity_bind(self._as_parameter_, arr)

        return np.array(arr, copy=True)

    def get_pressure(self):
        n = self._mesh.num_nodes
        arr = (ctypes.c_double * n)()
        _lib.flow_state_get_pressure_bind(self._as_parameter_, arr)

        return np.array(arr, copy=True)

    def __del__(self):
        if getattr(self, "_as_parameter_", None):
            _lib.flow_state_destroy_bind(self._as_parameter_)


class FlowState:
    """Holds velocity and pressure arrays for a mesh."""

    def __init__(self, mesh: Mesh):
        handle = _lib.flow_state_create_bind(mesh._as_parameter_)

        if not handle:
            raise MemoryError("Failed to create FlowState.")

        self._as_parameter_ = handle
        self._mesh = mesh

    def get_velocity(self):
        n = self._mesh.num_nodes
        arr = (ctypes.c_double * (3 * n))()
        _lib.flow_state_get_velocity_bind(self._as_parameter_, arr)

        return np.array(arr, copy=True)

    def get_pressure(self):
        n = self._mesh.num_nodes
        arr = (ctypes.c_double * n)()
        _lib.flow_state_get_pressure_bind(self._as_parameter_, arr)

        return np.array(arr, copy=True)

    def __del__(self):
        if getattr(self, "_as_parameter_", None):
            _lib.flow_state_destroy_bind(self._as_parameter_)


class Solver:
    """
    Wraps the solver life-cycle and provides a high-level simulation API.

    Binds to the following C functions (signatures shown for reference):
      void* solver_create_bind(void* mesh_handle, void* turb_model_handle);
      void  solver_initialize_bind(void* solver_handle);
      void  solver_apply_actuator_bind(void* solver_handle, void* actuator_handle, double dt);
      void  solver_step_bind(void* solver_handle, double dt);
      void  solver_read_state_bind(void* solver_handle, void* flow_state_handle);
      void  solver_destroy_bind(void* solver_handle);
    """

    def __init__(self, mesh: "Mesh", turb_model: "TurbulenceModel"):
        # --- Declare and check C signature for solver_create_bind ---
        _lib.solver_create_bind.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        _lib.solver_create_bind.restype = ctypes.c_void_p

        handle = _lib.solver_create_bind(mesh._as_parameter_, turb_model._as_parameter_)
        if not handle:
            raise MemoryError("[AeroBindings] Failed to create Solver.")
        self._as_parameter_ = handle
        self._mesh = mesh
        self._turb_model = turb_model

        print(
            f"[AeroBindings] Solver created successfully. (handle={hex(ctypes.addressof(ctypes.c_void_p(handle)))} )"
        )

    def initialize(self) -> None:
        """
        Initialize adjacency, discretization, and flow-state buffers on the C side.
        """
        _lib.solver_initialize_bind.argtypes = [ctypes.c_void_p]
        _lib.solver_initialize_bind.restype = None

        _lib.solver_initialize_bind(self._as_parameter_)
        print("[AeroBindings] Solver initialized.")

    def apply_actuator(self, actuator: "Actuator", dt: float) -> None:
        """
        Apply a single actuator’s influence for a timestep dt.
        :param actuator: an Actuator instance whose command is already set.
        :param dt: timestep in seconds (must be positive).
        """
        if dt <= 0:
            raise ValueError("[AeroBindings] dt must be positive for apply_actuator().")

        # Declare/verify C signature: (void*, void*, double) -> void
        _lib.solver_apply_actuator_bind.argtypes = [
            ctypes.c_void_p,  # solver handle
            ctypes.c_void_p,  # actuator handle
            ctypes.c_double,  # dt
        ]
        _lib.solver_apply_actuator_bind.restype = None

        _lib.solver_apply_actuator_bind(
            self._as_parameter_, actuator._as_parameter_, ctypes.c_double(dt)
        )
        print(f"[AeroBindings] Actuator applied to solver for dt={dt:.5f} seconds.")

    def step(self, dt: float) -> None:
        """
        Advance the solver by one timestep dt, updating internal solution state.
        :param dt: timestep in seconds (must be positive).
        """
        if dt <= 0:
            raise ValueError("[AeroBindings] dt must be positive for step().")

        # Declare/verify C signature: (void*, double) -> void
        _lib.solver_step_bind.argtypes = [
            ctypes.c_void_p,  # solver handle
            ctypes.c_double,  # dt
        ]
        _lib.solver_step_bind.restype = None

        _lib.solver_step_bind(self._as_parameter_, ctypes.c_double(dt))
        print(f"[AeroBindings] Solver stepped forward by dt={dt:.5f} seconds.")

    def read_state(self, flow_state: "FlowState") -> None:
        """
        Copy the solver’s internal flow state (velocity, pressure, etc.)
        into a FlowState object for inspection or post-processing.
        """
        # Declare/verify C signature: (void*, void*) -> void
        _lib.solver_read_state_bind.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        _lib.solver_read_state_bind.restype = None

        _lib.solver_read_state_bind(self._as_parameter_, flow_state._as_parameter_)
        print("[AeroBindings] Solver’s flow state copied into FlowState object.")

    def run(self, actuator: "Actuator", steps: int, dt: float) -> "FlowState":
        """
        Convenience method that:
          1) Initializes the solver
          2) Repeatedly applies the actuator and steps for a total of `steps` times
          3) Reads out the final flow state into a new FlowState object and returns it

        :param actuator: Actuator instance to use each step.
        :param steps: Number of timesteps to run (must be positive integer).
        :param dt: Timestep size in seconds (must be positive).
        :return: FlowState containing the final solution.
        """
        if steps <= 0:
            raise ValueError(
                "[AeroBindings] steps must be a positive integer for run()."
            )
        if dt <= 0:
            raise ValueError("[AeroBindings] dt must be positive for run().")

        print(f"[AeroBindings] Running solver for {steps} steps with dt={dt:.5f}.")

        # Initialize once before stepping
        self.initialize()

        for i in range(steps):
            print(f"[AeroBindings] Step {i+1}/{steps}...")
            self.apply_actuator(actuator, dt)
            self.step(dt)

        # Create output FlowState and read final solution
        out_state = FlowState(self._mesh)
        self.read_state(out_state)
        print("[AeroBindings] Completed run; returning final FlowState.")
        return out_state

    def __del__(self):
        """
        Destructor: ensure the C-level solver object is destroyed.
        """
        if getattr(self, "_as_parameter_", None):
            _lib.solver_destroy_bind.argtypes = [ctypes.c_void_p]
            _lib.solver_destroy_bind.restype = None

            _lib.solver_destroy_bind(self._as_parameter_)
            print("[AeroBindings] Solver destroyed.")
