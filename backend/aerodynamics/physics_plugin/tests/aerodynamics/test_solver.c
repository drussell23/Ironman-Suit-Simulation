// tests/aerodynamics/test_solver.c
// Tests for solver.c :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}

#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdlib.h>
#include <string.h>

#include "aerodynamics/solver.h"
#include "aerodynamics/mesh.h"
#include "aerodynamics/flow_state.h"
#include "aerodynamics/turbulence_model.h"
#include "aerodynamics/actuator.h"

//------------------------------------------------------------------------------
// 1) solver_create should reject NULL mesh or turb_model.
// ──────────────────────────────────────────────────────────────────────────────
static void test_solver_create_invalid(void **state)
{
    (void)state;
    Mesh *dummy_mesh = malloc(sizeof(Mesh));
    TurbulenceModel *dummy_tm = malloc(sizeof(TurbulenceModel));

    assert_null(solver_create(NULL, dummy_tm));
    assert_null(solver_create(dummy_mesh, NULL));

    free(dummy_mesh);
    free(dummy_tm);
}

//------------------------------------------------------------------------------
// 2) solver_create + solver_destroy basic integration.
// ──────────────────────────────────────────────────────────────────────────────
static void test_solver_create_and_destroy(void **state)
{
    (void)state;
    // minimal 1‐node mesh
    double coords[3] = {0.0, 0.0, 0.0};
    size_t conn1[1] = {0};
    Mesh *m = mesh_create(1, coords, 1, 1, conn1);
    TurbulenceModel *tm = malloc(sizeof(TurbulenceModel));

    Solver *s = solver_create(m, tm);
    assert_non_null(s);

    // cleanup; solver_destroy frees mesh & flow_state,
    // but not turb_model itself
    solver_destroy(s);
    free(tm);
}

//------------------------------------------------------------------------------
// 3) solver_read_state should copy zero‐initialized flow_state.
///  - Also no‐ops when solver or out is NULL.
// ──────────────────────────────────────────────────────────────────────────────
static void test_solver_read_state_copy(void **state)
{
    (void)state;
    double coords[3] = {0.0, 0.0, 0.0};
    size_t conn1[1] = {0};
    Mesh *m = mesh_create(1, coords, 1, 1, conn1);
    TurbulenceModel *tm = malloc(sizeof(TurbulenceModel));
    Solver *s = solver_create(m, tm);

    // prepare an output FlowState with allocated arrays
    FlowState out;
    size_t N = mesh_get_num_nodes(m);
    out.velocity = calloc(3 * N, sizeof(double));
    out.pressure = calloc(N, sizeof(double));
    out.turbulence_kinetic_energy = calloc(N, sizeof(double));
    out.turbulence_dissipation_rate = calloc(N, sizeof(double));

    // real flow_state is zero‐inited, so after copy we expect zeros
    solver_read_state(s, &out);
    assert_float_equal(out.velocity[0], 0.0, 1e-12);
    assert_float_equal(out.pressure[0], 0.0, 1e-12);

    free(out.velocity);
    free(out.pressure);
    free(out.turbulence_kinetic_energy);
    free(out.turbulence_dissipation_rate);

    solver_destroy(s);
    free(tm);
}

static void test_solver_read_state_null(void **state)
{
    (void)state;
    FlowState dummy;
    // should simply return without crashing
    solver_read_state(NULL, &dummy);
    solver_read_state((Solver *)1, NULL);
}

//------------------------------------------------------------------------------
// 4) solver_apply_actuators no‐ops on NULL inputs.
// ──────────────────────────────────────────────────────────────────────────────
static void test_solver_apply_actuators_null(void **state)
{
    (void)state;
    solver_apply_actuators(NULL, (Actuator *)1, 1, 0.1);

    // valid solver but NULL acts
    double coords[3] = {0.0, 0.0, 0.0};
    size_t conn1[1] = {0};
    Mesh *m = mesh_create(1, coords, 1, 1, conn1);
    TurbulenceModel *tm = malloc(sizeof(TurbulenceModel));
    Solver *s = solver_create(m, tm);

    solver_apply_actuators(s, NULL, 1, 0.1);

    solver_destroy(s);
    free(tm);
}

//------------------------------------------------------------------------------
// 5) solver_step should no‐op on NULL solver or non‐positive dt.
// ──────────────────────────────────────────────────────────────────────────────
static void test_solver_step_null(void **state)
{
    (void)state;
    solver_step(NULL, 0.1);

    double coords[3] = {0.0, 0.0, 0.0};
    size_t conn1[1] = {0};
    Mesh *m = mesh_create(1, coords, 1, 1, conn1);
    TurbulenceModel *tm = malloc(sizeof(TurbulenceModel));
    Solver *s = solver_create(m, tm);

    solver_step(s, 0.0);
    solver_destroy(s);
    free(tm);
}

//------------------------------------------------------------------------------
// 6) solver_initialize should safely handle NULL.
// ──────────────────────────────────────────────────────────────────────────────
static void test_solver_initialize_null(void **state)
{
    (void)state;
    // should simply return
    solver_initialize(NULL);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_solver_create_invalid),
        cmocka_unit_test(test_solver_create_and_destroy),
        cmocka_unit_test(test_solver_read_state_copy),
        cmocka_unit_test(test_solver_read_state_null),
        cmocka_unit_test(test_solver_apply_actuators_null),
        cmocka_unit_test(test_solver_step_null),
        cmocka_unit_test(test_solver_initialize_null),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
