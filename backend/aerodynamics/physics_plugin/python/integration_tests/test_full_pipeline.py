import unittest
import numpy as np
from bindings import Mesh, Solver, Actuator, TurbulenceModel, FlowState


class TestFullPipeline(unittest.TestCase):
    def test_full_pipeline_integration(self):
        # Create a simple tetrahedral mesh
        mesh = Mesh.tetrahedron(size=1.0)
        self.assertEqual(mesh.num_nodes, 4)

        # Set up turbulence model with standard k-epsilon parameters
        turb_model = TurbulenceModel()

        # Set up actuator on node 0, pushing in the x-direction
        actuator = Actuator(
            name="A0", node_ids=[0], direction=[1.0, 0.0, 0.0], gain=0.5
        )

        # Create solver and initialize
        solver = Solver(mesh, turb_model)
        solver.initialize()

        # Create flow state for reading intermediate results
        flow_state = FlowState(mesh)

        # Step through simulation, applying actuator at each step
        steps = 20
        dt = 0.05
        for step in range(steps):
            solver.apply_actuator(actuator, dt)
            solver.step(dt)

            # Read intermediate state and verify no NaNs at each step
            solver.read_state(flow_state)
            velocities = flow_state.get_velocity()
            pressures = flow_state.get_pressure()

            self.assertFalse(
                np.isnan(velocities).any(), f"Velocity contains NaNs at step {step}"
            )
            self.assertFalse(
                np.isnan(pressures).any(), f"Pressure contains NaNs at step {step}"
            )

        # Extract final simulation results
        final_velocities = flow_state.get_velocity()
        final_pressures = flow_state.get_pressure()

        # Optional sanity checks on final velocity magnitudes
        velocity_magnitude = np.linalg.norm(final_velocities.reshape(-1, 3), axis=1)
        self.assertTrue(
            np.all(velocity_magnitude >= 0),
            "Final velocity magnitudes must be non-negative",
        )

        # Check reasonable bounds on final pressures (placeholder example, adjust as needed)
        self.assertTrue(
            np.all(final_pressures >= 0), "Final pressures must be non-negative"
        )

        # Output for visual verification (can remove later)
        print("Final Velocities:", final_velocities)
        print("Final Pressures:", final_pressures)


if __name__ == "__main__":
    unittest.main()
