#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "aerodynamics/bindings.h"

// C99-compliant variadic LOG macro (handles one or more args)
#define LOG(...)                            \
    do                                      \
    {                                       \
        fprintf(stderr, "[AeroBindings] "); \
        fprintf(stderr, __VA_ARGS__);       \
        fprintf(stderr, "\n");              \
    } while (0)

#define CHECK_ALLOC(ptr, msg)                      \
    if (!(ptr))                                    \
    {                                              \
        LOG("Memory allocation failure: %s", msg); \
        return NULL;                               \
    }

// Mesh bindings implementation
MeshHandle mesh_create_bind(size_t num_nodes, const double *coords, size_t num_cells, size_t nodes_per_cell, const size_t *connectivity)
{
    if (!coords || !connectivity || num_nodes == 0 || num_cells == 0 || nodes_per_cell < 2)
    {
        LOG("Invalid mesh input parameters.");
        return NULL;
    }

    // Pass full topology to core API.
    Mesh *mesh = mesh_create(num_nodes, coords, num_cells, nodes_per_cell, connectivity);
    CHECK_ALLOC(mesh, "mesh structure");

    LOG("Mesh created successfully (nodes: %zu, cells: %zu)", num_nodes, num_cells);
    return (MeshHandle)mesh;
}

void mesh_destroy_bind(MeshHandle mesh)
{
    if (!mesh)
    {
        LOG("Attempted to destroy null mesh handle.");
        return;
    }
    mesh_destroy((Mesh *)mesh);
    LOG("Mesh destroyed.");
}

// Turbulence model bindings implementation
TurbulenceModelHandle turb_model_create_bind(double c_mu, double sigma_k, double sigma_eps, double c1_eps, double c2_eps)
{
    KEpsilonParameters params = {c_mu, sigma_k, sigma_eps, c1_eps, c2_eps};
    TurbulenceModel *model = turbulence_model_create(TURBULENCE_MODEL_K_EPSILON, &params);
    CHECK_ALLOC(model, "turbulence model");

    LOG("Turbulence model created (Cmu=%.3f, sigma_k=%.3f)", c_mu, sigma_k);
    return (TurbulenceModelHandle)model;
}

void turb_model_destroy_bind(TurbulenceModelHandle turb_model)
{
    if (!turb_model)
    {
        LOG("Attempted to destroy null turbulence model handle.");
        return;
    }
    turbulence_model_destroy((TurbulenceModel *)turb_model);
    LOG("Turbulence model destroyed.");
}

// Actuator bindings implementation
ActuatorHandle actuator_create_bind(const char *name, int type, size_t node_count, const size_t *node_ids, const double direction[3], double gain)
{
    if (!name || !node_ids || !direction || node_count == 0)
    {
        LOG("Invalid actuator input parameters.");
        return NULL;
    }

    Actuator *actuator = actuator_create(name, (ActuatorType)type, node_count, node_ids, direction, gain);
    CHECK_ALLOC(actuator, "actuator creation");

    LOG("Actuator '%s' created (type: %d, nodes: %zu)", name, type, node_count);
    return (ActuatorHandle)actuator;
}

void actuator_set_command_bind(ActuatorHandle act, double command)
{
    if (!act)
    {
        LOG("Attempted to set command on null actuator.");
        return;
    }
    actuator_set_command((Actuator *)act, command);
    LOG("Actuator command set to %.4f", command);
}

void actuator_destroy_bind(ActuatorHandle act)
{
    if (!act)
    {
        LOG("Attempted to destroy null actuator.");
        return;
    }
    actuator_destroy((Actuator *)act);
    LOG("Actuator destroyed.");
}

// Solver bindings implementation
SolverHandle solver_create_bind(MeshHandle mesh, TurbulenceModelHandle turb_model)
{
    if (!mesh || !turb_model)
    {
        LOG("Null input provided to solver_create.");
        return NULL;
    }

    Solver *solver = solver_create((Mesh *)mesh, (TurbulenceModel *)turb_model);
    CHECK_ALLOC(solver, "solver creation");

    LOG("Solver created successfully.");
    return (SolverHandle)solver;
}

void solver_initialize_bind(SolverHandle solver)
{
    if (!solver)
    {
        LOG("Attempted to initialize null solver.");
        return;
    }
    solver_initialize((Solver *)solver);
    LOG("Solver initialized.");
}

void solver_step_bind(SolverHandle solver, double dt)
{
    if (!solver || dt <= 0.0)
    {
        LOG("Invalid input to solver_step (solver: %p, dt: %f)", solver, dt);
        return;
    }
    solver_step((Solver *)solver, dt);
    LOG("Solver stepped by dt=%.5f seconds.", dt);
}

void solver_apply_actuator_bind(SolverHandle solver, ActuatorHandle actuator, double dt)
{
    if (!solver || !actuator || dt <= 0.0)
    {
        LOG("Invalid input to solver_apply_actuator "
            "(solver: %p, actuator: %p, dt: %f)",
            solver, actuator, dt);

        return;
    }

    // Forward the single ActuatorHandle into the core API
    solver_apply_actuators((Solver *)solver,
                           (const Actuator *)actuator,
                           1,
                           dt);

    LOG("Actuator applied to solver for dt=%.5f seconds.", dt);
}

void solver_destroy_bind(SolverHandle solver)
{
    if (!solver)
    {
        LOG("Attempted to destroy null solver.");
        return;
    }
    solver_destroy((Solver *)solver);
    LOG("Solver destroyed.");
}

// FlowState bindings implementation
FlowStateHandle flow_state_create_bind(MeshHandle mesh)
{
    if (!mesh)
    {
        LOG("Null mesh provided to flow_state_create.");
        return NULL;
    }
    FlowState *state = malloc(sizeof(FlowState));
    CHECK_ALLOC(state, "flow state allocation");

    if (flow_state_init(state, (Mesh *)mesh) != 0)
    {
        LOG("Flow state initialization failed.");
        free(state);
        return NULL;
    }
    LOG("Flow state created.");
    return (FlowStateHandle)state;
}

void solver_read_state_bind(SolverHandle solver, FlowStateHandle state)
{
    if (!solver || !state)
    {
        LOG("Invalid input to solver_read_state (solver: %p, state: %p)", solver, state);
        return;
    }
    solver_read_state((Solver *)solver, (FlowState *)state);
    LOG("Solver state read successfully.");
}

void flow_state_destroy_bind(FlowStateHandle state)
{
    if (!state)
    {
        LOG("Attempted to destroy null flow state.");
        return;
    }
    flow_state_destroy((FlowState *)state);
    free((FlowState *)state);
    LOG("Flow state destroyed.");
}

// Data extraction implementations with error checks
void flow_state_get_velocity_bind(FlowStateHandle state, double *out_velocity)
{
    FlowState *s = (FlowState *)state;
    if (!s || !out_velocity)
    {
        LOG("Invalid input to flow_state_get_velocity.");
        return;
    }
    memcpy(out_velocity, s->velocity, sizeof(double) * 3 * s->num_nodes);
}

void flow_state_get_pressure_bind(FlowStateHandle state, double *out_pressure)
{
    FlowState *s = (FlowState *)state;
    if (!s || !out_pressure)
    {
        LOG("Invalid input to flow_state_get_pressure.");
        return;
    }
    memcpy(out_pressure, s->pressure, sizeof(double) * s->num_nodes);
}

void flow_state_get_tke_bind(FlowStateHandle state, double *out_tke)
{
    FlowState *s = (FlowState *)state;
    if (!s || !out_tke)
    {
        LOG("Invalid input to flow_state_get_tke.");
        return;
    }
    memcpy(out_tke, s->turbulence_kinetic_energy, sizeof(double) * s->num_nodes);
}

void flow_state_get_dissipation_bind(FlowStateHandle state, double *out_eps)
{
    FlowState *s = (FlowState *)state;
    if (!s || !out_eps)
    {
        LOG("Invalid input to flow_state_get_dissipation.");
        return;
    }
    memcpy(out_eps, s->turbulence_dissipation_rate, sizeof(double) * s->num_nodes);
}
