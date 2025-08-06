"""
Improved Python bindings for the Aerodynamics Physics Plugin with proper error handling.
"""

import os
import ctypes
import numpy as np
from ctypes.util import find_library
from typing import List, Tuple, Optional
import warnings

# -----------------------------------------------------------------------------
# Load the shared library (cross-platform support)
# -----------------------------------------------------------------------------
_lib_name = "aerodynamics_physics_plugin"

def _find_library():
    """Find the aerodynamics physics plugin library."""
    # Try relative paths first
    base_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(base_dir, "..", "build", f"lib{_lib_name}.dylib"),  # macOS
        os.path.join(base_dir, "..", "build", f"lib{_lib_name}.so"),     # Linux
        os.path.join(base_dir, "..", "build", f"{_lib_name}.dll"),       # Windows
        os.path.join(base_dir, "..", "build", "Release", f"{_lib_name}.dll"),  # Windows Release
        os.path.join(base_dir, "..", "build", "Debug", f"{_lib_name}.dll"),    # Windows Debug
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # Fall back to system search
    found = find_library(_lib_name)
    if found:
        return found
    
    raise OSError(f"Could not find shared library '{_lib_name}'. "
                  f"Make sure the library is built and in the library path.")

try:
    _lib = ctypes.CDLL(_find_library())
except OSError as e:
    warnings.warn(f"Failed to load aerodynamics physics plugin: {e}")
    _lib = None

# -----------------------------------------------------------------------------
# Error handling
# -----------------------------------------------------------------------------
class AerodynamicsError(Exception):
    """Base exception for aerodynamics physics plugin errors."""
    pass

class AerodynamicsMemoryError(AerodynamicsError):
    """Raised when memory allocation fails."""
    pass

class AerodynamicsInvalidInputError(AerodynamicsError):
    """Raised when invalid input is provided."""
    pass

def _check_library_loaded():
    """Check if the library is loaded."""
    if _lib is None:
        raise AerodynamicsError("Aerodynamics physics plugin library is not loaded.")

def _check_null_result(result, error_msg="Operation failed"):
    """Check if a pointer result is NULL and raise exception if so."""
    if result is None or (isinstance(result, ctypes.c_void_p) and not result):
        raise AerodynamicsMemoryError(error_msg)
    return result

# -----------------------------------------------------------------------------
# Handle types
# -----------------------------------------------------------------------------
MeshHandle = ctypes.c_void_p
TurbulenceModelHandle = ctypes.c_void_p
ActuatorHandle = ctypes.c_void_p
SolverHandle = ctypes.c_void_p
FlowStateHandle = ctypes.c_void_p

# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------
class ActuatorType:
    SURFACE = 0
    BODY_FORCE = 1
    VOLUME = 2

class VTKFormat:
    ASCII = 0
    BINARY = 1
    XML = 2

# -----------------------------------------------------------------------------
# C function prototypes (only if library loaded)
# -----------------------------------------------------------------------------
if _lib is not None:
    # Mesh functions
    _lib.mesh_create_bind.restype = MeshHandle
    _lib.mesh_create_bind.argtypes = [
        ctypes.c_size_t,  # num_nodes
        ctypes.POINTER(ctypes.c_double),  # coords
        ctypes.c_size_t,  # num_cells
        ctypes.c_size_t,  # nodes_per_cell
        ctypes.POINTER(ctypes.c_size_t),  # connectivity
    ]
    _lib.mesh_destroy_bind.restype = None
    _lib.mesh_destroy_bind.argtypes = [MeshHandle]
    _lib.mesh_get_num_nodes.restype = ctypes.c_size_t
    _lib.mesh_get_num_nodes.argtypes = [MeshHandle]
    
    # Turbulence model functions
    _lib.turb_model_create_bind.restype = TurbulenceModelHandle
    _lib.turb_model_create_bind.argtypes = [
        ctypes.c_double,  # c_mu
        ctypes.c_double,  # sigma_k
        ctypes.c_double,  # sigma_eps
        ctypes.c_double,  # c1_eps
        ctypes.c_double,  # c2_eps
    ]
    _lib.turb_model_destroy_bind.restype = None
    _lib.turb_model_destroy_bind.argtypes = [TurbulenceModelHandle]
    
    # Actuator functions
    _lib.actuator_create_bind.restype = ActuatorHandle
    _lib.actuator_create_bind.argtypes = [
        ctypes.c_char_p,  # name
        ctypes.c_int,  # type
        ctypes.c_size_t,  # node_count
        ctypes.POINTER(ctypes.c_size_t),  # node_ids
        ctypes.POINTER(ctypes.c_double),  # direction
        ctypes.c_double,  # gain
    ]
    _lib.actuator_set_command_bind.restype = None
    _lib.actuator_set_command_bind.argtypes = [ActuatorHandle, ctypes.c_double]
    _lib.actuator_destroy_bind.restype = None
    _lib.actuator_destroy_bind.argtypes = [ActuatorHandle]
    
    # Solver functions
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
    _lib.solver_read_state_bind.restype = None
    _lib.solver_read_state_bind.argtypes = [SolverHandle, FlowStateHandle]
    
    # FlowState functions
    _lib.flow_state_create_bind.restype = FlowStateHandle
    _lib.flow_state_create_bind.argtypes = [MeshHandle]
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
    _lib.flow_state_get_tke_bind.restype = None
    _lib.flow_state_get_tke_bind.argtypes = [
        FlowStateHandle,
        ctypes.POINTER(ctypes.c_double),
    ]
    _lib.flow_state_get_dissipation_bind.restype = None
    _lib.flow_state_get_dissipation_bind.argtypes = [
        FlowStateHandle,
        ctypes.POINTER(ctypes.c_double),
    ]
    
    # VTK writer functions
    _lib.vtk_write_solution_bind.restype = ctypes.c_int
    _lib.vtk_write_solution_bind.argtypes = [
        ctypes.c_char_p,  # filename
        MeshHandle,       # mesh
        FlowStateHandle,  # state
        ctypes.c_int,     # format
    ]
    _lib.vtk_write_mesh_bind.restype = ctypes.c_int
    _lib.vtk_write_mesh_bind.argtypes = [
        ctypes.c_char_p,  # filename
        MeshHandle,       # mesh
        ctypes.c_int,     # format
    ]
    _lib.vtk_create_time_series_writer_bind.restype = ctypes.c_void_p
    _lib.vtk_create_time_series_writer_bind.argtypes = [
        ctypes.c_char_p,  # base_filename
        ctypes.c_int,     # format
    ]
    _lib.vtk_write_timestep_bind.restype = ctypes.c_int
    _lib.vtk_write_timestep_bind.argtypes = [
        ctypes.c_void_p,  # writer
        ctypes.c_int,     # timestep
        ctypes.c_double,  # time
        MeshHandle,       # mesh
        FlowStateHandle,  # state
    ]
    _lib.vtk_close_time_series_writer_bind.restype = None
    _lib.vtk_close_time_series_writer_bind.argtypes = [ctypes.c_void_p]

# -----------------------------------------------------------------------------
# Python wrapper classes with error handling
# -----------------------------------------------------------------------------
class Mesh:
    """Computational mesh for aerodynamics simulations."""
    
    def __init__(self, vertices: np.ndarray, cells: np.ndarray, nodes_per_cell: int = 4):
        """
        Create a mesh from vertices and cell connectivity.
        
        Args:
            vertices: Array of vertex coordinates, shape (n_vertices, 3)
            cells: Array of cell connectivity, shape (n_cells, nodes_per_cell)
            nodes_per_cell: Number of nodes per cell (default 4 for tetrahedra)
        """
        _check_library_loaded()
        
        # Validate inputs
        vertices = np.asarray(vertices, dtype=np.float64)
        cells = np.asarray(cells, dtype=np.uintp)
        
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise AerodynamicsInvalidInputError(
                f"Vertices must have shape (n_vertices, 3), got {vertices.shape}")
        
        if cells.ndim != 2 or cells.shape[1] != nodes_per_cell:
            raise AerodynamicsInvalidInputError(
                f"Cells must have shape (n_cells, {nodes_per_cell}), got {cells.shape}")
        
        # Flatten arrays for C interface
        n_vertices = vertices.shape[0]
        n_cells = cells.shape[0]
        coords_flat = vertices.flatten()
        cells_flat = cells.flatten()
        
        # Create C arrays
        coords_arr = (ctypes.c_double * len(coords_flat))(*coords_flat)
        cells_arr = (ctypes.c_size_t * len(cells_flat))(*cells_flat)
        
        # Create mesh
        handle = _lib.mesh_create_bind(
            n_vertices, coords_arr, n_cells, nodes_per_cell, cells_arr
        )
        self._handle = _check_null_result(handle, "Failed to create mesh")
        self._num_vertices = n_vertices
        self._num_cells = n_cells
        self._nodes_per_cell = nodes_per_cell
    
    @classmethod
    def tetrahedron(cls, size: float = 1.0):
        """Create a single tetrahedron mesh."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [size, 0.0, 0.0],
            [0.0, size, 0.0],
            [0.0, 0.0, size]
        ])
        cells = np.array([[0, 1, 2, 3]], dtype=np.uintp)
        return cls(vertices, cells, nodes_per_cell=4)
    
    @property
    def num_nodes(self) -> int:
        """Get number of nodes in the mesh."""
        if hasattr(self, '_handle') and self._handle:
            return _lib.mesh_get_num_nodes(self._handle)
        return self._num_vertices
    
    @property
    def num_cells(self) -> int:
        """Get number of cells in the mesh."""
        return self._num_cells
    
    def __del__(self):
        """Clean up mesh resources."""
        if hasattr(self, '_handle') and self._handle and _lib is not None:
            _lib.mesh_destroy_bind(self._handle)
            self._handle = None


class TurbulenceModel:
    """k-epsilon turbulence model."""
    
    def __init__(self, c_mu: float = 0.09, sigma_k: float = 1.0, 
                 sigma_eps: float = 1.3, c1_eps: float = 1.44, c2_eps: float = 1.92):
        """
        Create a k-epsilon turbulence model.
        
        Args:
            c_mu: Model constant (default 0.09)
            sigma_k: Turbulent Prandtl number for k (default 1.0)
            sigma_eps: Turbulent Prandtl number for epsilon (default 1.3)
            c1_eps: Model constant (default 1.44)
            c2_eps: Model constant (default 1.92)
        """
        _check_library_loaded()
        
        handle = _lib.turb_model_create_bind(c_mu, sigma_k, sigma_eps, c1_eps, c2_eps)
        self._handle = _check_null_result(handle, "Failed to create turbulence model")
        self.c_mu = c_mu
        self.sigma_k = sigma_k
        self.sigma_eps = sigma_eps
        self.c1_eps = c1_eps
        self.c2_eps = c2_eps
    
    def __del__(self):
        """Clean up turbulence model resources."""
        if hasattr(self, '_handle') and self._handle and _lib is not None:
            _lib.turb_model_destroy_bind(self._handle)
            self._handle = None


class Actuator:
    """Force/momentum actuator for flow control."""
    
    def __init__(self, name: str, node_ids: List[int], direction: List[float], 
                 gain: float = 1.0, actuator_type: int = ActuatorType.BODY_FORCE):
        """
        Create an actuator.
        
        Args:
            name: Name of the actuator
            node_ids: List of node IDs where actuator applies forces
            direction: Force direction vector [x, y, z]
            gain: Force gain/scaling factor (default 1.0)
            actuator_type: Type of actuator (default BODY_FORCE)
        """
        _check_library_loaded()
        
        # Validate inputs
        if not name:
            raise AerodynamicsInvalidInputError("Actuator name cannot be empty")
        
        if not node_ids:
            raise AerodynamicsInvalidInputError("Node IDs list cannot be empty")
        
        direction = np.asarray(direction, dtype=np.float64)
        if direction.shape != (3,):
            raise AerodynamicsInvalidInputError(
                f"Direction must be a 3D vector, got shape {direction.shape}")
        
        # Normalize direction
        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-10:
            raise AerodynamicsInvalidInputError("Direction vector cannot be zero")
        direction = direction / dir_norm
        
        # Create C arrays
        name_b = name.encode('utf-8')
        n_nodes = len(node_ids)
        nodes_arr = (ctypes.c_size_t * n_nodes)(*node_ids)
        dir_arr = (ctypes.c_double * 3)(*direction)
        
        # Create actuator
        handle = _lib.actuator_create_bind(
            name_b, actuator_type, n_nodes, nodes_arr, dir_arr, gain
        )
        self._handle = _check_null_result(handle, f"Failed to create actuator '{name}'")
        self.name = name
        self.node_ids = list(node_ids)
        self.direction = direction
        self.gain = gain
        self.actuator_type = actuator_type
        self._command = 0.0
    
    def set_command(self, command: float):
        """Set actuator command/strength."""
        if not hasattr(self, '_handle') or not self._handle:
            raise AerodynamicsError("Actuator has been destroyed")
        
        _lib.actuator_set_command_bind(self._handle, command)
        self._command = command
    
    @property
    def command(self) -> float:
        """Get current actuator command."""
        return self._command
    
    def __del__(self):
        """Clean up actuator resources."""
        if hasattr(self, '_handle') and self._handle and _lib is not None:
            _lib.actuator_destroy_bind(self._handle)
            self._handle = None


class FlowState:
    """Flow field state (velocity, pressure, turbulence)."""
    
    def __init__(self, mesh: Mesh):
        """
        Create a flow state for the given mesh.
        
        Args:
            mesh: The mesh to create flow state for
        """
        _check_library_loaded()
        
        if not hasattr(mesh, '_handle') or not mesh._handle:
            raise AerodynamicsInvalidInputError("Invalid mesh provided")
        
        handle = _lib.flow_state_create_bind(mesh._handle)
        self._handle = _check_null_result(handle, "Failed to create flow state")
        self._mesh = mesh
    
    def get_velocity(self) -> np.ndarray:
        """Get velocity field as numpy array of shape (n_nodes, 3)."""
        if not hasattr(self, '_handle') or not self._handle:
            raise AerodynamicsError("Flow state has been destroyed")
        
        n = self._mesh.num_nodes
        arr = (ctypes.c_double * (3 * n))()
        _lib.flow_state_get_velocity_bind(self._handle, arr)
        
        return np.array(arr, copy=True).reshape((n, 3))
    
    def get_pressure(self) -> np.ndarray:
        """Get pressure field as numpy array of shape (n_nodes,)."""
        if not hasattr(self, '_handle') or not self._handle:
            raise AerodynamicsError("Flow state has been destroyed")
        
        n = self._mesh.num_nodes
        arr = (ctypes.c_double * n)()
        _lib.flow_state_get_pressure_bind(self._handle, arr)
        
        return np.array(arr, copy=True)
    
    def get_tke(self) -> np.ndarray:
        """Get turbulent kinetic energy field as numpy array of shape (n_nodes,)."""
        if not hasattr(self, '_handle') or not self._handle:
            raise AerodynamicsError("Flow state has been destroyed")
        
        n = self._mesh.num_nodes
        arr = (ctypes.c_double * n)()
        _lib.flow_state_get_tke_bind(self._handle, arr)
        
        return np.array(arr, copy=True)
    
    def get_dissipation(self) -> np.ndarray:
        """Get turbulent dissipation rate field as numpy array of shape (n_nodes,)."""
        if not hasattr(self, '_handle') or not self._handle:
            raise AerodynamicsError("Flow state has been destroyed")
        
        n = self._mesh.num_nodes
        arr = (ctypes.c_double * n)()
        _lib.flow_state_get_dissipation_bind(self._handle, arr)
        
        return np.array(arr, copy=True)
    
    def __del__(self):
        """Clean up flow state resources."""
        if hasattr(self, '_handle') and self._handle and _lib is not None:
            _lib.flow_state_destroy_bind(self._handle)
            self._handle = None


class Solver:
    """CFD solver for aerodynamics simulations."""
    
    def __init__(self, mesh: Mesh, turbulence_model: TurbulenceModel):
        """
        Create a solver.
        
        Args:
            mesh: Computational mesh
            turbulence_model: Turbulence model to use
        """
        _check_library_loaded()
        
        if not hasattr(mesh, '_handle') or not mesh._handle:
            raise AerodynamicsInvalidInputError("Invalid mesh provided")
        
        if not hasattr(turbulence_model, '_handle') or not turbulence_model._handle:
            raise AerodynamicsInvalidInputError("Invalid turbulence model provided")
        
        handle = _lib.solver_create_bind(mesh._handle, turbulence_model._handle)
        self._handle = _check_null_result(handle, "Failed to create solver")
        self._mesh = mesh
        self._turbulence_model = turbulence_model
        self._initialized = False
        self._time = 0.0
        self._step_count = 0
    
    def initialize(self):
        """Initialize the solver (must be called before stepping)."""
        if not hasattr(self, '_handle') or not self._handle:
            raise AerodynamicsError("Solver has been destroyed")
        
        _lib.solver_initialize_bind(self._handle)
        self._initialized = True
    
    def step(self, dt: float):
        """
        Advance solution by one time step.
        
        Args:
            dt: Time step size in seconds
        """
        if not hasattr(self, '_handle') or not self._handle:
            raise AerodynamicsError("Solver has been destroyed")
        
        if not self._initialized:
            raise AerodynamicsError("Solver must be initialized before stepping")
        
        if dt <= 0:
            raise AerodynamicsInvalidInputError(f"Time step must be positive, got {dt}")
        
        _lib.solver_step_bind(self._handle, dt)
        self._time += dt
        self._step_count += 1
    
    def apply_actuator(self, actuator: Actuator, dt: float):
        """
        Apply actuator forces for the given time step.
        
        Args:
            actuator: Actuator to apply
            dt: Time step size in seconds
        """
        if not hasattr(self, '_handle') or not self._handle:
            raise AerodynamicsError("Solver has been destroyed")
        
        if not hasattr(actuator, '_handle') or not actuator._handle:
            raise AerodynamicsInvalidInputError("Invalid actuator provided")
        
        if dt <= 0:
            raise AerodynamicsInvalidInputError(f"Time step must be positive, got {dt}")
        
        _lib.solver_apply_actuator_bind(self._handle, actuator._handle, dt)
    
    def read_state(self, flow_state: FlowState):
        """
        Read current flow state into the provided FlowState object.
        
        Args:
            flow_state: FlowState object to read into
        """
        if not hasattr(self, '_handle') or not self._handle:
            raise AerodynamicsError("Solver has been destroyed")
        
        if not hasattr(flow_state, '_handle') or not flow_state._handle:
            raise AerodynamicsInvalidInputError("Invalid flow state provided")
        
        _lib.solver_read_state_bind(self._handle, flow_state._handle)
    
    @property
    def time(self) -> float:
        """Get current simulation time."""
        return self._time
    
    @property
    def step_count(self) -> int:
        """Get number of steps taken."""
        return self._step_count
    
    def __del__(self):
        """Clean up solver resources."""
        if hasattr(self, '_handle') and self._handle and _lib is not None:
            _lib.solver_destroy_bind(self._handle)
            self._handle = None


# -----------------------------------------------------------------------------
# Convenience functions
# -----------------------------------------------------------------------------
def create_tetrahedral_mesh(vertices: np.ndarray, tetrahedra: np.ndarray) -> Mesh:
    """
    Create a tetrahedral mesh from vertices and tetrahedra connectivity.
    
    Args:
        vertices: Array of vertex coordinates, shape (n_vertices, 3)
        tetrahedra: Array of tetrahedra connectivity, shape (n_tets, 4)
    
    Returns:
        Mesh object
    """
    return Mesh(vertices, tetrahedra, nodes_per_cell=4)

def create_hexahedral_mesh(vertices: np.ndarray, hexahedra: np.ndarray) -> Mesh:
    """
    Create a hexahedral mesh from vertices and hexahedra connectivity.
    
    Args:
        vertices: Array of vertex coordinates, shape (n_vertices, 3)
        hexahedra: Array of hexahedra connectivity, shape (n_hexes, 8)
    
    Returns:
        Mesh object
    """
    return Mesh(vertices, hexahedra, nodes_per_cell=8)


# -----------------------------------------------------------------------------
# VTK writer wrapper classes
# -----------------------------------------------------------------------------
class VTKWriter:
    """Write mesh and solution data to VTK files for visualization."""
    
    @staticmethod
    def write_solution(filename: str, mesh: Mesh, flow_state: FlowState, 
                      format: int = VTKFormat.XML) -> None:
        """
        Write mesh and flow state to VTK file.
        
        Args:
            filename: Output filename (without extension)
            mesh: Mesh object
            flow_state: FlowState object
            format: VTK file format (default XML)
        
        Raises:
            AerodynamicsError: If write fails
        """
        _check_library_loaded()
        
        if not hasattr(mesh, '_handle') or not mesh._handle:
            raise AerodynamicsInvalidInputError("Invalid mesh provided")
        
        if not hasattr(flow_state, '_handle') or not flow_state._handle:
            raise AerodynamicsInvalidInputError("Invalid flow state provided")
        
        filename_b = filename.encode('utf-8')
        result = _lib.vtk_write_solution_bind(
            filename_b, mesh._handle, flow_state._handle, format
        )
        
        if result != 0:
            raise AerodynamicsError(f"Failed to write VTK file: {filename}")
    
    @staticmethod
    def write_mesh(filename: str, mesh: Mesh, 
                   format: int = VTKFormat.XML) -> None:
        """
        Write mesh only to VTK file.
        
        Args:
            filename: Output filename (without extension)
            mesh: Mesh object
            format: VTK file format (default XML)
        
        Raises:
            AerodynamicsError: If write fails
        """
        _check_library_loaded()
        
        if not hasattr(mesh, '_handle') or not mesh._handle:
            raise AerodynamicsInvalidInputError("Invalid mesh provided")
        
        filename_b = filename.encode('utf-8')
        result = _lib.vtk_write_mesh_bind(filename_b, mesh._handle, format)
        
        if result != 0:
            raise AerodynamicsError(f"Failed to write VTK mesh file: {filename}")


class VTKTimeSeriesWriter:
    """Write time series data for transient simulations."""
    
    def __init__(self, base_filename: str, format: int = VTKFormat.XML):
        """
        Create a time series writer.
        
        Args:
            base_filename: Base filename for series
            format: VTK file format (default XML)
        """
        _check_library_loaded()
        
        filename_b = base_filename.encode('utf-8')
        handle = _lib.vtk_create_time_series_writer_bind(filename_b, format)
        self._handle = _check_null_result(handle, "Failed to create VTK time series writer")
        self.base_filename = base_filename
        self.format = format
        self._timestep_count = 0
    
    def write_timestep(self, timestep: int, time: float, 
                      mesh: Mesh, flow_state: FlowState) -> None:
        """
        Write a timestep to the series.
        
        Args:
            timestep: Timestep number
            time: Physical time
            mesh: Mesh object
            flow_state: FlowState object
        
        Raises:
            AerodynamicsError: If write fails
        """
        if not hasattr(self, '_handle') or not self._handle:
            raise AerodynamicsError("Time series writer has been closed")
        
        if not hasattr(mesh, '_handle') or not mesh._handle:
            raise AerodynamicsInvalidInputError("Invalid mesh provided")
        
        if not hasattr(flow_state, '_handle') or not flow_state._handle:
            raise AerodynamicsInvalidInputError("Invalid flow state provided")
        
        result = _lib.vtk_write_timestep_bind(
            self._handle, timestep, time, mesh._handle, flow_state._handle
        )
        
        if result != 0:
            raise AerodynamicsError(f"Failed to write timestep {timestep}")
        
        self._timestep_count += 1
    
    def close(self) -> None:
        """Close the time series writer."""
        if hasattr(self, '_handle') and self._handle:
            _lib.vtk_close_time_series_writer_bind(self._handle)
            self._handle = None
    
    def __del__(self):
        """Ensure writer is closed on deletion."""
        self.close()
    
    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.close()

__all__ = [
    'AerodynamicsError',
    'AerodynamicsMemoryError', 
    'AerodynamicsInvalidInputError',
    'ActuatorType',
    'VTKFormat',
    'Mesh',
    'TurbulenceModel',
    'Actuator',
    'FlowState',
    'Solver',
    'VTKWriter',
    'VTKTimeSeriesWriter',
    'create_tetrahedral_mesh',
    'create_hexahedral_mesh',
]