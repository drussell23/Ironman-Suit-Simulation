# backend/aerodynamics/environmental_effects/turbulence_models/__init__.py

"""
Turbulence‐models subpackage.

Provides:
  – Two‐equation k–ε model (in k_epsilon.py)
  – Static and dynamic Smagorinsky LES model (in smagorinsky.py)
"""

from __future__ import annotations

# —— k–ε model exports ——
from .k_epsilon import (
    # constants
    C_MU,
    SIGMA_K,
    SIGMA_EPS,
    C1_EPS,
    C2_EPS,
    PRANDTL_T,
    GRAVITY,
    KOLMOGOROV_CONSTANT,
    
    # Core API
    enforce_positive,
    turbulent_viscosity,
    turbulent_diffusivity,
    dissipation_rate_equilibrium,
    production_tensor,
    production_rate,
    buoyancy_production,
    wall_damping_function,
    near_wall_damping,
    advance_transport,
    eddy_lifetime,
    kolmogorov_length_scale,
    taylor_microscale,
    turbulence_intensity,
)

# —— Smagorinsky LES model exports ——
from .smagorinsky import (
    C_S,
    filter_width,
    strain_rate_tensor,
    strain_rate_magnitude,
    smagorinsky_viscosity,
    subgrid_stress_tensor,
    dynamic_smagorinsky_constant,
)

__all__ = [
    # k–ε
    "C_MU",
    "SIGMA_K",
    "SIGMA_EPS",
    "C1_EPS",
    "C2_EPS",
    "PRANDTL_T",
    "GRAVITY",
    "KOLMOGOROV_CONSTANT",
    "enforce_positive",
    "turbulent_viscosity",
    "turbulent_diffusivity",
    "dissipation_rate_equilibrium",
    "production_tensor",
    "production_rate",
    "buoyancy_production",
    "wall_damping_function",
    "near_wall_damping",
    "advance_transport",
    "eddy_lifetime",
    "kolmogorov_length_scale",
    "taylor_microscale",
    "turbulence_intensity",
    # Smagorinsky
    "C_S",
    "filter_width",
    "strain_rate_tensor",
    "strain_rate_magnitude",
    "smagorinsky_viscosity",
    "subgrid_stress_tensor",
    "dynamic_smagorinsky_constant",
]
