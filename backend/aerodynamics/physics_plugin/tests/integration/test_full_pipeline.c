#include <assert.h>

#include "aerodynamics/mesh.h"
#include "aerodynamics/solver.h"
#include "aerodynamics/actuator.h"
#include "aerodynamics/flow_state.h"
#include "aerodynamics/turbulence_model.h"

int main(void) {
    // 1) Make a 4-node tetrahedral mesh.
    size_t num_nodes = 4;

    double coords[4 * 3] = {
        0.0, 0.0, 0.0, // node 0
        1.0, 0.0, 0.0, // node 1
        0.0, 1.0, 0.0, // node 2
        0.0, 0.0, 1.0  // node 3
    };

    size_t num_cells = 1;
    size_t nodes_per_cell = 4;
    size_t connectivity[4] = {0, 1, 2, 3};

    Mesh *mesh = mesh_create(num_nodes, coords, num_cells, nodes_per_cell, connectivity);
    assert(mesh);

    mesh->num_cells = num_cells;
    mesh->nodes_per_cell = nodes_per_cell;
    mesh->connectivity = malloc(sizeof(connectivity));

    memcpy(mesh->connectivity, connectivity, sizeof(connectivity));

    // 2) Create turbulence model (k-epsilon).
    TurbulenceModel *tm = turbulence_model_create(TURBULENCE_MODEL_K_EPSILON, NULL);
    assert(tm);

    // 3) Create & initialize solver.
    Solver *solver = solver_create(mesh, tm);
    assert(solver);
    solver_initialize(solver);

    // 4) Create one surface actuator on node 0.
    size_t act_nodes[1] = {0};
    double dir[3] = {1.0, 0.0, 0.0};
    Actuator *act = actuator_create("A0", ACTUATOR_TYPE_SURFACE, 1, act_nodes, dir, 1.0);
    assert(act);
    actuator_set_command(act, 0.5);

    // 5) Step the full pipeline.
    const double dt = 0.05;

    for (int step = 0; step < 20; ++step) {
        solver_apply_actuators(solver, act, 1, dt);
        solver_step(solver, dt);
    }

    // 6) Read back final state.
    FlowState *out = malloc(sizeof(*out));
    assert(out);
    assert(flow_state_init(out, mesh) == 0);
    solver_read_state(solver, out);

    // 7) Sanity: check: no NaNs in velocity or pressure.
    size_t N = mesh_get_num_nodes(mesh);

    for (size_t i = 0; i < N; ++i) {
        for (int d = 0; d < 3; ++d) {
            assert(!isnan(out->velocity[3 * i + d]));
        }
        assert(!isnan(out->pressure[i]));
    }

    // 8) Cleanup.
    flow_state_destroy(out);
    free(out);
    actuator_destroy(act);
    solver_destroy(solver);
    turbulence_model_destroy(tm);
    mesh_destroy(mesh);

    return 0;
}