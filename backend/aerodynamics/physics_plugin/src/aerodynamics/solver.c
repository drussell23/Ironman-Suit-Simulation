// backend/aerodynamics/environmental_effects/physics_plugin/src/solver.c

#include <stdlib.h>  // for malloc, free
#include <stdio.h>   // for printf

#include "aerodynamics/solver.h"
#include "aerodynamics/mesh.h"
#include "aerodynamics/turbulence_model.h"
#include "aerodynamics/flow_state.h"
#include "aerodynamics/actuator.h"

// Opaque struct definition. 
struct Solver {
    Mesh *mesh;               // Pointer to the mesh
    TurbulenceModel *turb_model; // Pointer to the turbulence model
    FlowState *flow_state;      // Pointer to the flow state
};

/// Create a new Solver instance.
Solver *solver_create(Mesh *mesh, TurbulenceModel *turb_model) {
    // Allocate memory for the Solver struct.
    if (!mesh || !turb_model) {
        fprintf(stderr, "solver_create: mesh or turb_model is NULL.\n");
        return NULL; // Invalid input
    }

    Solver *solver = (Solver *)malloc(sizeof(Solver)); // Allocate memory for Solver struct.

    if (!solver) {
        fprintf(stderr, "solver_create: Memory allocation failed.\n");
        return NULL; // Memory allocation failure
    }

    solver->mesh = mesh; // Set mesh pointer.
    solver->turb_model = turb_model; // Set turbulence model pointer.
    solver->flow_state = (FlowState*)malloc(sizeof(FlowState)); // Allocate memory for FlowState struct.    

    // Check for allocation failure.
    if (!solver->flow_state) {
        fprintf(stderr, "solver_create: Memory allocation failed for flow_state.\n");
        free(solver); // Free previously allocated memory.
        return NULL; // Memory allocation failure
    }

    // Initialize the flow state.
    if (flow_state_init(solver->flow_state, mesh) != 0) {
        fprintf(stderr, "solver_create: flow_state_init failed.\n");
        free(solver->flow_state); // Free previously allocated memory.
        free(solver); // Free Solver struct.
        return NULL; // Initialization failure
    }

    return solver; // Return the created Solver object.
}

// Destroy the Solver instance.
void solver_destroy(Solver *solver) {
    // Free the Solver instance and its internal resources.
    if (!solver) {
        return; // Nothing to destroy
    }

    flow_state_destroy(solver->flow_state); // Free FlowState struct.
    free(solver->flow_state); // Free FlowState struct.
    free(solver->mesh); // Free Mesh struct.
}

// Initialize the solver's internal resources.
void solver_initialize(Solver *solver) {
    // Initialize the solver's internal resources.
    if (!solver) {
        return; // Nothing to initialize
    }

    // Initialize the solver's internal resources.
    if (mesh_initialize(solver->mesh) != 0) {
        fprintf(stderr, "solver_initialize: Mesh initalization failed.\n");
    }

    turb_model_initialize(solver->turb_model, solver->mesh, solver->flow_state); // Initialize turbulence model.
}

// Read the current flow state into the user-provided structure.
void solver_read_state(const Solver *solver, FlowState *out) {
    // Read the current flow state into the user-provided structure.
    if (!solver || !out) {
        return; // Nothing to read
    }

    size_t N = mesh_get_num_nodes(solver->mesh); // Get number of nodes from mesh.

    memcpy(out->velocity, solver->flow_state->velocity, sizeof(double) * 3 * N); // Copy velocity.
    memcpy(out->pressure, solver->flow_state->pressure, sizeof(double) * N); // Copy pressure.
    memcpy(out->turbulence_kinetic_energy, solver->flow_state->turbulence_kinetic_energy, sizeof(double) * N); // Copy turbulence kinetic energy.
    memcpy(out->turbulence_dissipation_rate, solver->flow_state->turbulence_dissipation_rate, sizeof(double) * N); // Copy turbulence dissipation rate. 
}

// Apply actuator commands to the solver.
void solver_apply_actuators(Solver *solver, const Actuator *acts, size_t act_count, double dt) {
    // Apply actuator commands to the solver.
    if (!solver || !acts) {
        return; // Nothing to apply
    }

    for (size_t idx = 0; idx < act_count; ++idx) {
        actuator_apply(&acts[idx], solver->mesh, solver->flow_state, dt); // Apply actuator effect.
    }
}

// Advance the solution by one time step.
void solver_step(Solver *solver, double dt) {
    // Advance the solution by one time step.
    if (!solver || dt <= 0.0) {
        return; // Nothing to step
    }

    // Step 1: Convection update.
    mesh_compute_convection(solver->mesh, solver->flow_state, dt); // Compute convection term.

    // Step 2: Turbulent viscosity computation.
    double *nu_t = turb_model_compute_visocsity(solver->turb_model, solver->mesh, solver->flow_state); // Compute turbulent viscosity.

    if (!nu_t) {
        fprintf(stderr, "solver_step: Turbulent viscosity computation failed.\n");
        return; // Memory allocation failure
    }

    // Step 3: Diffusion step (using nu_t from turbulence model).
    mesh_compute_diffusion(solver->mesh, solver->flow_state, nu_t, dt); // Compute diffusion term.

    free(nu_t); // Free turbulent viscosity array.

    // Step 4: Update turbulence model (k, epsilon) fields.
    turb_model_update(solver->turb_model, solver->mesh, solver->flow_state, dt); // Update turbulence model.
}
