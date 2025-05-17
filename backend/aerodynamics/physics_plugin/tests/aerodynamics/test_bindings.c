// File: physics_plugin/tests/test_bindings.c

#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include "aerodynamics/bindings.h"
#include "aerodynamics/mesh.h"
#include "aerodynamics/flow_state.h"
#include "aerodynamics/turbulence_model.h"
#include "aerodynamics/actuator.h"
#include "aerodynamics/solver.h"

/**
 * We’ll build a trivial 2-node, 1-cell mesh for most tests:
 *   coords = { (0,0,0), (1,0,0) }
 *   connectivity = {0,1}
 */
static const double sample_coords[6] = {
    0.0, 0.0, 0.0,
    1.0, 0.0, 0.0};
static const size_t sample_conn[2] = {0, 1};

static void test_mesh_bind_null_and_bad_params(void **state)
{
    (void)state;
    /* Null coords or connectivity, or zero counts, or too few nodes_per_cell */
    assert_null(mesh_create_bind(0, sample_coords, 1, 2, sample_conn));
    assert_null(mesh_create_bind(2, NULL, 1, 2, sample_conn));
    assert_null(mesh_create_bind(2, sample_coords, 0, 2, sample_conn));
    assert_null(mesh_create_bind(2, sample_coords, 1, 1, sample_conn));
    assert_null(mesh_create_bind(2, sample_coords, 1, 2, NULL));
}

static void test_mesh_bind_create_and_destroy(void **state)
{
    (void)state;
    MeshHandle mh = mesh_create_bind(
        2, sample_coords, 1, 2, sample_conn);
    assert_non_null(mh);

    Mesh *m = (Mesh *)mh;
    assert_int_equal(m->num_nodes, 2);
    assert_int_equal(m->num_cells, 1);
    assert_int_equal(m->nodes_per_cell, 2);

    /* destroy twice: second call must be safe no‐op */
    mesh_destroy_bind(mh);
    mesh_destroy_bind(NULL);
}

static void test_turb_model_bind_create_and_destroy(void **state)
{
    (void)state;
    /* Bind layer never rejects by params, so just test create/destroy safety */
    TurbulenceModelHandle th = turb_model_create_bind(
        0.09, 1.0, 1.3, 1.44, 1.92);
    assert_non_null(th);

    turb_model_destroy_bind(th);
    turb_model_destroy_bind(NULL);
}

static void test_actuator_bind_null_and_bad_params(void **state)
{
    (void)state;
    size_t nodes[2] = {0, 1};
    double dir[3] = {1., 0., 0.};

    assert_null(actuator_create_bind(
        NULL, ACTUATOR_TYPE_SURFACE, 2, nodes, dir, 1.0));
    assert_null(actuator_create_bind(
        "a", ACTUATOR_TYPE_SURFACE, 0, nodes, dir, 1.0));
    assert_null(actuator_create_bind(
        "a", ACTUATOR_TYPE_SURFACE, 2, NULL, dir, 1.0));
    assert_null(actuator_create_bind(
        "a", ACTUATOR_TYPE_SURFACE, 2, nodes, NULL, 1.0));
}

static void test_actuator_bind_create_set_destroy(void **state)
{
    (void)state;
    size_t nodes[2] = {0, 1};
    double dir[3] = {0., 1., 0.};

    ActuatorHandle ah = actuator_create_bind(
        "ctrl", ACTUATOR_TYPE_BODY_FORCE,
        2, nodes, dir, 2.5);
    assert_non_null(ah);

    /* setting on valid and NULL must not crash */
    actuator_set_command_bind(ah, 3.1415);
    actuator_set_command_bind(NULL, 1.0);

    actuator_destroy_bind(ah);
    actuator_destroy_bind(NULL);
}

static void test_solver_and_flowstate_bindings(void **state)
{
    (void)state;
    /* Build mesh + model first */
    MeshHandle mh = mesh_create_bind(2, sample_coords, 1, 2, sample_conn);
    TurbulenceModelHandle th = turb_model_create_bind(0.09, 1.0, 1.3, 1.44, 1.92);

    /* solver_create rejects on NULL args */
    assert_null(solver_create_bind(NULL, th));
    assert_null(solver_create_bind(mh, NULL));

    SolverHandle sh = solver_create_bind(mh, th);
    assert_non_null(sh);

    /* initialize, step, apply actuator: invalid calls are no-ops */
    solver_initialize_bind(NULL);
    solver_initialize_bind(sh);

    solver_step_bind(NULL, 0.01);
    solver_step_bind(sh, -1.0);
    solver_step_bind(sh, 0.05);

    /* flow‐state */
    FlowStateHandle fh_null = flow_state_create_bind(NULL);
    assert_null(fh_null);

    FlowStateHandle fh = flow_state_create_bind(mh);
    assert_non_null(fh);

    /* reading state */
    solver_read_state_bind(NULL, fh);
    solver_read_state_bind(sh, NULL);
    solver_read_state_bind(sh, fh);

    /* data‐extraction buffers */
    double vel[2 * 3], pres[2], tke[2], eps[2];

    flow_state_get_velocity_bind(NULL, vel);
    flow_state_get_velocity_bind(fh, NULL);
    flow_state_get_velocity_bind(fh, vel);

    flow_state_get_pressure_bind(fh, pres);
    flow_state_get_tke_bind(fh, tke);
    flow_state_get_dissipation_bind(fh, eps);

    /* actuator via solver binding (1 actuator) */
    size_t anodes[2] = {0, 1};
    double adir[3] = {0, 0, 1};
    ActuatorHandle ah = actuator_create_bind(
        "surf", ACTUATOR_TYPE_SURFACE, 2, anodes, adir, 1.0);
    solver_apply_actuator_bind(NULL, ah, 0.1);
    solver_apply_actuator_bind(sh, NULL, 0.1);
    solver_apply_actuator_bind(sh, ah, -0.5);
    solver_apply_actuator_bind(sh, ah, 1.0);

    /* cleanup all */
    flow_state_destroy_bind(fh);
    solver_destroy_bind(sh);
    actuator_destroy_bind(ah);
    turb_model_destroy_bind(th);
    mesh_destroy_bind(mh);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_mesh_bind_null_and_bad_params),
        cmocka_unit_test(test_mesh_bind_create_and_destroy),
        cmocka_unit_test(test_turb_model_bind_create_and_destroy),
        cmocka_unit_test(test_actuator_bind_null_and_bad_params),
        cmocka_unit_test(test_actuator_bind_create_set_destroy),
        cmocka_unit_test(test_solver_and_flowstate_bindings),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
