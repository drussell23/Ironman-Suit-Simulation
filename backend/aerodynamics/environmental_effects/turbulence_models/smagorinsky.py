"""
Advanced Smagorinsky Subgrid-Scale (SGS) Model for Large Eddy Simulation (LES)

Implements enhanced Smagorinsky SGS modeling with both static and dynamic approaches:
- Computation of LES filter width (Δ) based on grid spacing
- Eddy viscosity ν_t calculation (static and dynamic Smagorinsky)
- Strain-rate tensor S_ij
- Strain-rate magnitude |S|
- Subgrid-scale stress tensor τ_ij = -2 ν_t S_ij
- Robust dynamic Smagorinsky constant calculation using Germano identity with stability and adaptivity enhancements

Ideal for high-fidelity aerodynamic and turbulent flow simulations, especially suitable for advanced aerodynamic applications like humanoid jet propulsion.
"""

from __future__ import annotations
from typing import Union, Tuple, TypeAlias
import numpy as np

# Smagorinsky model default constant (typical range: 0.1–0.2)
C_S: float = 0.17

Number: TypeAlias = Union[float, int]
ArrayLike: TypeAlias = Union[Number, np.ndarray]


def filter_width(dx: ArrayLike, dy: ArrayLike, dz: ArrayLike) -> np.ndarray:
    dx, dy, dz = map(lambda x: np.asarray(x, float), (dx, dy, dz))
    if np.any(dx <= 0) or np.any(dy <= 0) or np.any(dz <= 0):
        raise ValueError("Grid spacings must be positive")
    return (dx * dy * dz) ** (1.0 / 3.0)


def strain_rate_tensor(grad_u: ArrayLike) -> np.ndarray:
    grad = np.asarray(grad_u, float)
    if grad.shape[-2:] != (3, 3):
        raise ValueError("grad_u must have shape (..., 3, 3)")
    return 0.5 * (grad + grad.swapaxes(-2, -1))


def strain_rate_magnitude(grad_u: ArrayLike) -> np.ndarray:
    S = strain_rate_tensor(grad_u)
    return np.sqrt(2.0 * np.sum(S * S, axis=(-2, -1)))


def smagorinsky_viscosity(
        grad_u: ArrayLike,
        dx: ArrayLike,
        dy: ArrayLike,
        dz: ArrayLike,
        C_s: float = C_S) -> np.ndarray:
    Δ = filter_width(dx, dy, dz)
    S_mag = strain_rate_magnitude(grad_u)
    return (C_s * Δ) ** 2 * S_mag


def subgrid_stress_tensor(
        grad_u: ArrayLike,
        dx: ArrayLike,
        dy: ArrayLike,
        dz: ArrayLike,
        C_s: float = C_S) -> np.ndarray:
    ν_t = smagorinsky_viscosity(grad_u, dx, dy, dz, C_s)
    S = strain_rate_tensor(grad_u)
    τ = -2.0 * ν_t[..., np.newaxis, np.newaxis] * S
    return τ


def dynamic_smagorinsky_constant(
        grad_u: ArrayLike,
        grad_tilde_u: ArrayLike,
        dx: ArrayLike,
        dy: ArrayLike,
        dz: ArrayLike,
        Cs_bar: float = C_S,
        relaxation_factor: float = 0.6,
        stability_epsilon: float = 1e-8) -> np.ndarray:
    """
    Advanced Dynamic Smagorinsky model: computes local Cs adaptively via Germano identity.

    Parameters:
        grad_u: resolved-scale velocity gradient tensor
        grad_tilde_u: test-filtered velocity gradient tensor
        dx, dy, dz: grid spacings
        Cs_bar: initial Smagorinsky constant guess
        relaxation_factor: relaxation factor for numerical stability (0-1)
        stability_epsilon: small number to prevent division by zero

    Returns:
        Adaptive local Cs values array
    """
    Δ = filter_width(dx, dy, dz)
    Δ_tilde = 2 * Δ

    S = strain_rate_tensor(grad_u)
    S_tilde = strain_rate_tensor(grad_tilde_u)

    S_mag = strain_rate_magnitude(grad_u)
    S_tilde_mag = strain_rate_magnitude(grad_tilde_u)

    # Leonard stress tensor
    L_ij = Δ_tilde**2 * S_tilde * S_tilde_mag - Δ**2 * S * S_mag

    # Germano identity tensors
    M_ij = 2 * (Δ**2 * (S_tilde_mag * S_tilde - S_mag * S))

    # Local calculation of Cs^2
    numerator = np.sum(L_ij * M_ij, axis=(-2, -1))
    denominator = np.sum(M_ij * M_ij, axis=(-2, -1)) + stability_epsilon

    Cs_local_squared = numerator / denominator
    Cs_local_squared = np.clip(Cs_local_squared, 0, None)

    Cs_local = np.sqrt(Cs_local_squared)

    # Stability and smoothing via relaxation
    Cs_local_smoothed = relaxation_factor * Cs_local + (1 - relaxation_factor) * Cs_bar

    return Cs_local_smoothed
