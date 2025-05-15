import sys
import os
import unittest
import numpy as np

from backend.aerodynamics.environmental_effects.turbulence_models import k_epsilon as ke


class TestTurbulenceModel(unittest.TestCase):

    def test_turbulent_viscosity_scalar(self):
        k = 0.4
        eps = 0.2
        expected = ke.C_MU * k**2 / eps
        result = ke.turbulent_viscosity(k, eps)
        self.assertAlmostEqual(result, expected, places=6)

    def test_turbulent_diffusivity_scalar(self):
        nu_t = 0.03
        Pr_t = 0.9
        expected = nu_t / Pr_t
        result = ke.turbulent_diffusivity(nu_t, Pr_t)
        self.assertAlmostEqual(result, expected, places=6)

    def test_dissipation_rate_equilibrium_scalar(self):
        k = 0.5
        L = 0.1
        expected = (ke.C_MU**0.75) * k**1.5 / L
        result = ke.dissipation_rate_equilibrium(k, L)
        self.assertAlmostEqual(result, expected, places=6)

    def test_production_tensor_shape(self):
        grad = np.eye(3)
        result = ke.production_tensor(grad)
        self.assertEqual(result.shape, (3, 3))

    def test_production_rate_scalar(self):
        grad = np.eye(3)
        k = 0.4
        eps = 0.1
        result = ke.production_rate(grad, k, eps)
        self.assertIsInstance(result, float)

    def test_buoyancy_production_scalar(self):
        beta = 0.003
        g = np.array([0, -9.81, 0])
        grad_T = np.array([0, 1.0, 0])
        nu_t = 0.01
        result = ke.buoyancy_production(beta, g, grad_T, nu_t)
        self.assertIsInstance(result, float)

    def test_wall_damping_output(self):
        y_plus = np.array([5.0, 10.0, 20.0])
        result = ke.wall_damping_function(y_plus)
        self.assertTrue(np.all(result >= 0))

    def test_near_wall_damping_scalar(self):
        nu_t = 0.02
        y = 0.005
        result = ke.near_wall_damping(nu_t, y)
        self.assertGreaterEqual(result, 0.0)

    def test_advance_transport_returns_positive(self):
        k = 0.3
        eps = 0.2
        prod = 0.05
        buoy = 0.01
        dt = 0.01
        k_new, eps_new = ke.advance_transport(k, eps, prod, buoy, dt)
        self.assertGreater(k_new, 0)
        self.assertGreater(eps_new, 0)

    def test_eddy_lifetime(self):
        k = 0.3
        eps = 0.1
        expected = k / eps
        result = ke.eddy_lifetime(k, eps)
        self.assertAlmostEqual(result, expected, places=6)

    def test_kolmogorov_length_scale(self):
        nu = 1e-5
        eps = 0.1
        result = ke.kolmogorov_length_scale(nu, eps)
        self.assertGreater(result, 0.0)

    def test_taylor_microscale(self):
        k = 0.4
        eps = 0.2
        nu = 1e-5
        result = ke.taylor_microscale(k, eps, nu)
        self.assertGreater(result, 0.0)

    def test_turbulence_intensity(self):
        k = 0.2
        U = np.array([10.0, 0.0, 0.0])
        result = ke.turbulence_intensity(k, U)
        self.assertGreater(result, 0.0)


if __name__ == "__main__":
    unittest.main()
