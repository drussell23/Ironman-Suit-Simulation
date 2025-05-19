// -----------------------------------------------------------------------------
// solver.c
// -----------------------------------------------------------------------------
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "aerodynamics/solver.h"
#include "aerodynamics/mesh.h"
#include "aerodynamics/turbulence_model.h"
#include "aerodynamics/flow_state.h"
#include "aerodynamics/actuator.h"

// Internal solver struct.
struct Solver
{
    Mesh *mesh;
    TurbulenceModel *turb_model;
    FlowState *flow_state;
};

Solver *solver_create(Mesh *mesh, TurbulenceModel *turb_model)
{
    if (!mesh || !turb_model)
    {
        fprintf(stderr, "solver_create: NULL mesh or turb_model.\n");
        return NULL;
    }
    Solver *s = malloc(sizeof(*s));
    if (!s)
    {
        fprintf(stderr, "solver_create: malloc failed.\n");
        return NULL;
    }
    s->mesh = mesh;
    s->turb_model = turb_model;
    s->flow_state = NULL;
    return s;
}

void solver_destroy(Solver *solver)
{
    if (!solver)
        return;
    if (solver->flow_state)
    {
        flow_state_destroy(solver->flow_state);
        free(solver->flow_state);
    }
    free(solver);
}

void solver_initialize(Solver *solver)
{
    if (!solver || !solver->mesh || !solver->turb_model)
    {
        fprintf(stderr, "solver_initialize: invalid input.\n");
        return;
    }

    // 1) Build mesh connectivity, volumes, adjacency, etc.
    if (mesh_initialize(solver->mesh) != 0)
    {
        fprintf(stderr, "solver_initialize: mesh init failed.\n");
        return;
    }

    // 2) Allocate and zero-init the FlowState
    solver->flow_state = malloc(sizeof(*solver->flow_state));
    if (!solver->flow_state)
    {
        fprintf(stderr, "solver_initialize: flow_state malloc failed.\n");
        return;
    }
    if (flow_state_init(solver->flow_state, solver->mesh) != 0)
    {
        fprintf(stderr, "solver_initialize: flow_state_init failed.\n");
        free(solver->flow_state);
        solver->flow_state = NULL;
        return;
    }

    // 3) Initialize turbulence-model fields (k, ε)
    turb_model_initialize(solver->turb_model,
                          solver->mesh,
                          solver->flow_state);
}

void solver_read_state(const Solver *solver, FlowState *out)
{
    if (!solver || !out || !solver->flow_state)
        return;

    size_t N = mesh_get_num_nodes(solver->mesh);

    // Copy arrays.
    memcpy(out->velocity, solver->flow_state->velocity, sizeof(double) * 3 * N);
    memcpy(out->pressure, solver->flow_state->pressure, sizeof(double) * N);
    memcpy(out->turbulence_kinetic_energy, solver->flow_state->turbulence_kinetic_energy, sizeof(double) * N);
    memcpy(out->turbulence_dissipation_rate, solver->flow_state->turbulence_dissipation_rate, sizeof(double) * N);
}

void solver_apply_actuators(Solver *solver,
                            const Actuator *acts,
                            size_t act_count,
                            double dt)
{
    if (!solver || !acts || act_count == 0 || dt <= 0.0)
        return;

    if (!solver->flow_state)
        return;

    Mesh *mesh = solver->mesh;
    FlowState *state = solver->flow_state;
    
    for (size_t i = 0; i < act_count; ++i)
    {
        actuator_apply(&acts[i], mesh, state, dt);
    }
}

void solver_step(Solver *solver, double dt)
{
    if (!solver || dt <= 0.0 || !solver->flow_state)
        return;
    // Convection
    mesh_compute_convection(solver->mesh, solver->flow_state, dt);
    // Turbulent viscosity
    double *nu_t = turb_model_compute_viscosity(solver->turb_model,
                                                solver->mesh,
                                                solver->flow_state);
    if (!nu_t)
    {
        fprintf(stderr, "solver_step: viscosity computation failed.\n");
        return;
    }
    // Diffusion
    mesh_compute_diffusion(solver->mesh, solver->flow_state, nu_t, dt);
    free(nu_t);
    // Turbulence update
    turb_model_update(solver->turb_model, solver->mesh, solver->flow_state, dt);
}
