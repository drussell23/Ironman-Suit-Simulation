#!/usr/bin/env python3
"""
VTK output demonstration for the aerodynamics physics plugin.

This example shows how to:
1. Create a simple mesh
2. Run a simulation
3. Write results to VTK files for ParaView visualization
"""

import sys
import os
import numpy as np

# Add the plugin's python directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from bindings_improved import (
    Mesh, TurbulenceModel, Solver, Actuator, FlowState,
    VTKWriter, VTKTimeSeriesWriter, 
    ActuatorType, VTKFormat
)

def create_box_mesh(nx=5, ny=5, nz=5, size=1.0):
    """Create a simple box mesh for testing."""
    # Create vertices
    x = np.linspace(0, size, nx)
    y = np.linspace(0, size, ny)
    z = np.linspace(0, size, nz)
    
    vertices = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                vertices.append([x[i], y[j], z[k]])
    
    vertices = np.array(vertices)
    
    # Create tetrahedral cells (simplified - just for demo)
    # In a real case, you'd use a proper meshing library
    cells = []
    for k in range(nz-1):
        for j in range(ny-1):
            for i in range(nx-1):
                # Get the 8 corners of a cube
                n000 = i + j*nx + k*nx*ny
                n100 = (i+1) + j*nx + k*nx*ny
                n010 = i + (j+1)*nx + k*nx*ny
                n110 = (i+1) + (j+1)*nx + k*nx*ny
                n001 = i + j*nx + (k+1)*nx*ny
                n101 = (i+1) + j*nx + (k+1)*nx*ny
                n011 = i + (j+1)*nx + (k+1)*nx*ny
                n111 = (i+1) + (j+1)*nx + (k+1)*nx*ny
                
                # Split cube into 6 tetrahedra
                cells.extend([
                    [n000, n100, n010, n001],
                    [n100, n110, n010, n111],
                    [n010, n110, n011, n111],
                    [n001, n101, n011, n111],
                    [n000, n001, n010, n011],
                    [n100, n101, n001, n111]
                ])
    
    cells = np.array(cells, dtype=np.uintp)
    
    return Mesh(vertices, cells, nodes_per_cell=4)

def main():
    print("=== VTK Output Demonstration ===")
    
    # Create a simple box mesh
    print("\n1. Creating mesh...")
    mesh = create_box_mesh(nx=8, ny=8, nz=8, size=2.0)
    print(f"   Mesh created with {mesh.num_nodes} nodes")
    
    # Write mesh only (for inspection)
    print("\n2. Writing mesh to VTK...")
    VTKWriter.write_mesh("output/mesh_only", mesh, format=VTKFormat.XML)
    VTKWriter.write_mesh("output/mesh_only_legacy", mesh, format=VTKFormat.ASCII)
    
    # Set up simulation
    print("\n3. Setting up simulation...")
    turb_model = TurbulenceModel()
    solver = Solver(mesh, turb_model)
    solver.initialize()
    
    # Create actuators (thrusters)
    print("\n4. Creating actuators...")
    # Find nodes near the bottom for thrust
    coords = mesh.coordinates
    bottom_nodes = []
    for i in range(mesh.num_nodes):
        if coords[3*i + 2] < 0.1:  # z < 0.1
            bottom_nodes.append(i)
    
    if bottom_nodes:
        actuator = Actuator(
            name="bottom_thruster",
            node_ids=bottom_nodes[:20],  # Use first 20 bottom nodes
            direction=[0.0, 0.0, 1.0],   # Upward thrust
            gain=100.0,
            actuator_type=ActuatorType.BODY_FORCE
        )
        actuator.set_command(0.8)  # 80% thrust
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Single timestep output
    print("\n5. Running single timestep...")
    dt = 0.001
    solver.apply_actuator(actuator, dt)
    solver.step(dt)
    
    # Read solution
    flow_state = FlowState(mesh)
    solver.read_state(flow_state)
    
    # Write single solution
    print("\n6. Writing single solution to VTK...")
    VTKWriter.write_solution("output/solution_single", mesh, flow_state, 
                           format=VTKFormat.XML)
    
    # Time series simulation
    print("\n7. Running time series simulation...")
    n_steps = 50
    output_interval = 5
    
    # Reset solver
    solver = Solver(mesh, turb_model)
    solver.initialize()
    
    # Create time series writer
    with VTKTimeSeriesWriter("output/solution_series", format=VTKFormat.XML) as writer:
        for step in range(n_steps):
            # Apply forces and step
            solver.apply_actuator(actuator, dt)
            solver.step(dt)
            
            # Output every N steps
            if step % output_interval == 0:
                solver.read_state(flow_state)
                writer.write_timestep(step, step * dt, mesh, flow_state)
                
                # Get some statistics
                velocity = flow_state.get_velocity()
                v_mag = np.sqrt(np.sum(velocity**2, axis=1))
                print(f"   Step {step:3d}: max velocity = {np.max(v_mag):.3e} m/s")
    
    print("\n8. VTK files written to 'output/' directory")
    print("   - mesh_only.vtu: Mesh structure only")
    print("   - mesh_only_legacy.vtk: Legacy VTK format")
    print("   - solution_single.vtu: Single timestep solution")
    print("   - solution_series.pvd: Time series (open in ParaView)")
    print("\nTo visualize:")
    print("   1. Install ParaView: https://www.paraview.org/download/")
    print("   2. Open ParaView")
    print("   3. File -> Open -> Select .vtu or .pvd file")
    print("   4. Click 'Apply' in Properties panel")
    print("   5. Use 'Glyph' filter to visualize velocity vectors")
    print("   6. Use 'Contour' filter to visualize pressure isosurfaces")

if __name__ == "__main__":
    main()