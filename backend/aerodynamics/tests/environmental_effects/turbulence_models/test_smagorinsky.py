# backend/aerodynamics/tests/environmental_effects/turbulence_models/test_smagorinsky.py

import sys
import os
import unittest
import numpy as np

# Make sure the package root is on PYTHONPATH
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")),
)

from backend.aerodynamics.environmental_effects.turbulence_models import (
    smagorinsky as sm,
)


class TestSmagorinsky(unittest.TestCase):
    def test_filter_width_scalar(self):
        dx, dy, dz = 0.1, 0.2, 0.3
        expected = (dx * dy * dz) ** (1 / 3)
        self.assertAlmostEqual(sm.filter_width(dx, dy, dz), expected, places=7)

    def test_filter_width_array(self):
        dx = np.array([0.1, 0.2])
        dy = np.array([0.2, 0.3])
        dz = np.array([0.3, 0.4])
        expected = (dx * dy * dz) ** (1 / 3)
        np.testing.assert_allclose(sm.filter_width(dx, dy, dz), expected, rtol=1e-7)

    def test_strain_rate_tensor_invalid_shape(self):
        with self.assertRaises(ValueError):
            sm.strain_rate_tensor(np.zeros((2, 2)))

    def test_strain_rate_tensor_symmetry(self):
        grad = np.array([[0, 1, 2], [3, 0, 4], [5, 6, 0]], float)
        S = sm.strain_rate_tensor(grad)
        # Must be symmetric
        self.assertTrue(np.allclose(S, S.T), "Strain rate tensor is not symmetric")

    def test_strain_rate_magnitude_shear(self):
        # grad[0,1]=1 → S[0,1]=S[1,0]=0.5 → sum(S^2)=0.5 → |S|=sqrt(2*0.5)=1.0
        grad = np.zeros((3, 3), float)
        grad[0, 1] = 1.0
        mag = sm.strain_rate_magnitude(grad)
        expected = np.sqrt(2.0 * (0.5**2 + 0.5**2))
        self.assertAlmostEqual(mag, expected, places=7)

    def test_smagorinsky_viscosity(self):
        grad = np.zeros((3, 3), float)
        grad[0, 1] = 1.0
        # Δ=1, |S|=1 → ν_t = (C_s * Δ)^2 * |S|
        nu_t = sm.smagorinsky_viscosity(grad, 1.0, 1.0, 1.0, C_s=0.2)
        self.assertAlmostEqual(nu_t, (0.2 * 1.0) ** 2 * 1.0, places=7)

    def test_subgrid_stress_tensor_shape_and_sign(self):
        dx = dy = dz = 1.0
        C_s = 0.2
        grad = np.zeros((3, 3), float)
        grad[0, 1] = 1.0
        τ = sm.subgrid_stress_tensor(grad, dx, dy, dz, C_s=C_s)
        # symmetric tensor, and off-diagonal must be negative
        self.assertEqual(τ.shape, (3, 3))
        self.assertTrue(np.allclose(τ, τ.T))
        self.assertLess(τ[0, 1], 0.0)

    def test_dynamic_smagorinsky_constant_uniform_gradient(self):
        """
        When grad == grad_tilde, the Germano numerator is zero ⇒
        local Cs^2 = 0 ⇒ Cs_local = sqrt(0) blended with Cs_bar ⇒
        0 <= Cs_local < Cs_bar
        """
        grad = np.zeros((3, 3), float)
        grad[0, 1] = 1.0
        dx = dy = dz = 1.0
        Cs_bar = 0.17

        Cs_local = sm.dynamic_smagorinsky_constant(
            grad, grad, dx, dy, dz, Cs_bar=Cs_bar
        )

        # Should remain non‐negative and strictly less than Cs_bar
        self.assertGreaterEqual(Cs_local, 0.0)
        self.assertLess(Cs_local, Cs_bar)


if __name__ == "__main__":
    unittest.main()
