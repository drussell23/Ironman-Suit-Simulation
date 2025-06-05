/*
 * File: examples/full_pipeline_example.c
 * Extended aerodynamic pipeline example in C:
 *   - Command-line args for dt and steps
 *   - Mesh creation, connectivity & adjacency dump
 *   - FlowState init & intermediate diagnostics
 *   - k–ε turbulence model init & viscosity print
 *   - Two actuators (surface + body-force)
 *   - Solver init, time stepping with per-step avg velocity/pressure
 *   - Simulation timing
 *   - Final state print and cleanup
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "aerodynamics/mesh.h"
#include "aerodynamics/flow_state.h"
#include "aerodynamics/turbulence_model.h"
#include "aerodynamics/actuator.h"
#include "aerodynamics/solver.h"

/**
 * Compute average velocity magnitude at all nodes in the flow state.
 *
 * @param state flow state with node velocity data
 * @return average velocity magnitude
 */
static double compute_avg_velocity(const FlowState *state)
{
    double sum = 0.0;
    for (size_t i = 0; i < state->num_nodes; ++i)
    {
        double vx = state->velocity[3 * i];
        double vy = state->velocity[3 * i + 1];
        double vz = state->velocity[3 * i + 2];
        sum += sqrt(vx * vx + vy * vy + vz * vz);
    }
    return sum / state->num_nodes;
}

/**
 * Compute average pressure at all nodes in the flow state.
 *
 * @param state flow state with node pressure data
 * @return average pressure
 */
static double compute_avg_pressure(const FlowState *state)
{
    double sum = 0.0; // Initialize sum to zero. 
    
    // Iterate through all nodes and accumulate pressure values.
    // Note: state->pressure is assumed to be an array of size num_nodes. 
    for (size_t i = 0; i < state->num_nodes; ++i)
        sum += state->pressure[i];
    return sum / state->num_nodes;
}

/**
 * Print the usage information for the program.
 *
 * @param prog The name of the program, typically argv[0].
 * 
 * The usage information includes the expected command-line arguments:
 * - dt: time step size (default 0.1)
 * - steps: number of timesteps (default 10)
 */

static void print_usage(const char *prog)
{
    fprintf(stderr, "Usage: %s [dt] [steps]\n", prog);
    fprintf(stderr, "  dt: time step size (default 0.1)\n");
    fprintf(stderr, "  steps: number of timesteps (default 10)\n");
}

/**
 * Extended Aerodynamics Pipeline Example
 *
 * This example demonstrates the full pipeline of the Aerodynamics physics plugin,
 * including:
 *  1) Mesh creation
 *  2) FlowState initialization
 *  3) Turbulence model (k-epsilon) initialization
 *  4) Actuator creation (wing & thruster)
 *  5) Solver initialization
 *  6) Time stepping with per-step diagnostics
 *  7) Final state print
 *  8) Cleanup
 *
 * The example takes two optional command-line arguments:
 *  - dt: time step size (default 0.1)
 *  - steps: number of timesteps (default 10)
 *
 * Usage: ./full_pipeline_example [dt] [steps]
 *
 * @return 0 on success, nonzero on error
 */
int main(int argc, char **argv)
{
    // Command-line args
    double dt = 0.1;
    int steps = 10;
    
    if (argc > 1) {}
        dt = atof(argv[1]);
    if (argc > 2)
        steps = atoi(argv[2]);
    if (steps < 1)
    {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    printf("Extended Aerodynamics Pipeline Example\n");
    printf(" Time step dt = %.4f, steps = %d\n\n", dt, steps);

    // 1) Mesh creation: simple tetrahedron
    size_t num_nodes = 4;
    double coords[12] = {
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0};
    size_t num_cells = 1;
    size_t nodes_per_cell = 4;
    size_t connectivity[4] = {0, 1, 2, 3};

    Mesh *mesh = mesh_create(num_nodes, coords, num_cells, nodes_per_cell, connectivity);
    if (!mesh)
    {
        fprintf(stderr, "Error: mesh_create failed\n");
        return EXIT_FAILURE;
    }
    if (mesh_initialize(mesh))
    {
        fprintf(stderr, "Error: mesh_initialize failed\n");
        mesh_destroy(mesh);
        return EXIT_FAILURE;
    }

    printf("Mesh: %zu nodes, %zu cells\n", mesh_get_num_nodes(mesh), mesh->num_cells);
    // Print connectivity and volumes
    for (size_t c = 0; c < mesh->num_cells; ++c)
    {
        printf(" Cell %zu: nodes [", c);
        for (size_t j = 0; j < nodes_per_cell; ++j)
            printf("%zu%s", mesh->connectivity[c * nodes_per_cell + j], j < nodes_per_cell - 1 ? ", " : "");
        printf("] , volume = %g\n", mesh->cell_volumes[c]);
    }
    // Print adjacency offsets
    printf(" Mesh adjacency offsets: ");
    for (size_t i = 0; i <= mesh->num_cells; ++i)
        printf("%zu ", mesh->adjacency_offsets[i]);
    printf("\n\n");

    // 2) FlowState init
    FlowState state; // Initialize flow state with mesh.

    if (flow_state_init(&state, mesh))
    {
        fprintf(stderr, "Error: flow_state_init failed\n");
        mesh_destroy(mesh);
        return EXIT_FAILURE; // Return failure if flow state initialization fails.
    }
    printf("Initialized flow state (vel & press zeroed)\n\n");

    // 3) Turbulence model (k-epsilon)
    KEpsilonParameters params = {0.09, 1.0, 1.3, 1.44, 1.92}; // Default values for k-epsilon.
    TurbulenceModel *turb = turbulence_model_create(TURBULENCE_MODEL_K_EPSILON, &params);
    if (!turb)
    {
        fprintf(stderr, "Error: turbulence_model_create failed\n");
        flow_state_destroy(&state);
        mesh_destroy(mesh);
        return EXIT_FAILURE; // Return failure if turbulence model creation fails.
    }
    turb_model_initialize(turb, mesh, &state);
    printf("Turbulence model initialized (k-epsilon)\n\n");

    // Initial eddy viscosity
    double *nu_t0 = turb_model_compute_viscosity(turb, mesh, &state);
    printf("Initial eddy viscosity at nodes:");
    for (size_t i = 0; i < state.num_nodes; ++i)
        printf(" %g", nu_t0[i]);
    printf("\n\n");
    free(nu_t0);

    // 4) Actuators
    size_t surf_ids[] = {0, 1};
    double surf_dir[3] = {0.0, 1.0, 0.0};
    Actuator *surf = actuator_create("wing", ACTUATOR_TYPE_SURFACE, 2, surf_ids, surf_dir, 0.5);
    size_t bf_ids[] = {2, 3};
    double bf_dir[3] = {1.0, 0.0, 0.0};
    Actuator *bf = actuator_create("thruster", ACTUATOR_TYPE_BODY_FORCE, 2, bf_ids, bf_dir, 2.0);
    if (!surf || !bf)
    {
        fprintf(stderr, "Error: actuator_create failed\n");
        actuator_destroy(surf);
        actuator_destroy(bf);
        turbulence_model_destroy(turb);
        flow_state_destroy(&state);
        mesh_destroy(mesh);
        return EXIT_FAILURE;
    }
    printf("Created actuators: wing(surface), thruster(body-force)\n\n");

    // 5) Solver init
    Solver *solver = solver_create(mesh, turb);
    if (!solver)
    {
        fprintf(stderr, "Error: solver_create failed\n");
        actuator_destroy(surf);
        actuator_destroy(bf);
        turbulence_model_destroy(turb);
        flow_state_destroy(&state);
        mesh_destroy(mesh);
        return EXIT_FAILURE;
    }
    solver_initialize(solver);
    printf("Solver initialized\n\n");

    // 6) Time stepping
    time_t t_start = time(NULL);
    for (int step = 0; step < steps; ++step)
    {
        double cmd_w = 0.1 * step;
        double cmd_t = 0.2 * step;
        actuator_set_command(surf, cmd_w);
        actuator_set_command(bf, cmd_t);
        Actuator acts[2] = {*surf, *bf};
        solver_apply_actuators(solver, acts, 2, dt);
        solver_step(solver, dt);

        // per-step diagnostics
        FlowState tmp;
        flow_state_init(&tmp, mesh);
        solver_read_state(solver, &tmp);
        double avgv = compute_avg_velocity(&tmp);
        double avgp = compute_avg_pressure(&tmp);
        printf(" Step %d: cmd(wing=%.2f thr=%.2f), avg_vel=%.3f, avg_pres=%.3f\n", step, cmd_w, cmd_t, avgv, avgp);
        flow_state_destroy(&tmp);
    }
    double elapsed = difftime(time(NULL), t_start);
    printf("\nCompleted %d steps in %.0f seconds\n\n", steps, elapsed);

    // 7) Final state
    FlowState final;
    flow_state_init(&final, mesh);
    solver_read_state(solver, &final);
    printf("Final velocities & pressures:\n");
    for (size_t i = 0; i < final.num_nodes; ++i)
        printf(" Node %zu: vel=(%.4f,%.4f,%.4f), p=%.4f\n", i, final.velocity[3 * i], final.velocity[3 * i + 1], final.velocity[3 * i + 2], final.pressure[i]);
    flow_state_destroy(&final);

    // 8) Cleanup
    solver_destroy(solver);
    actuator_destroy(surf);
    actuator_destroy(bf);
    turbulence_model_destroy(turb);
    flow_state_destroy(&state);
    mesh_destroy(mesh);
}