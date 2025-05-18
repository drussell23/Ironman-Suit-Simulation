// tests/aerodynamics/test_turbulence_model.c
// Tests for turbulence_model.c  :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}

#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "aerodynamics/turbulence_model.h"
#include "aerodynamics/mesh.h"
#include "aerodynamics/flow_state.h"

// 1) Creation and destruction
static void test_turbulence_model_create_default(void **state)
{
    (void)state;

    // Create with NULL params: should get default k-eps values
    TurbulenceModel *m = turbulence_model_create(TURBULENCE_MODEL_K_EPSILON, NULL);

    assert_non_null(m);
    assert_int_equal(m->type, TURBULENCE_MODEL_K_EPSILON);

    // Default constants.
    assert_true(fabs(m->params.c_mu - 0.09) < 1e-12);
    assert_true(fabs(m->params.sigma_k - 1.0) < 1e-12);
    assert_true(fabs(m->params.sigma_eps - 1.3) < 1e-12);
    assert_true(fabs(m->params.c1_eps - 1.44) < 1e-12);
    assert_true(fabs(m->params.c2_eps - 1.92) < 1e-12);

    turbulence_model_destroy(m);
}

static void test_turbulence_model_create_with_params(void **state)
{
    (void)state;

    KEpsilonParameters params = {
        .c_mu = 0.1, .sigma_k = 2.0, .sigma_eps = 3.0, .c1_eps = 4.0, .c2_eps = 5.0};

    TurbulenceModel *m = turbulence_model_create(TURBULENCE_MODEL_K_EPSILON, &params);

    assert_non_null(m);

    assert_true(fabs(m->params.c_mu - 0.1) < 1e-12);
    assert_true(fabs(m->params.sigma_k - 2.0) < 1e-12);
    assert_true(fabs(m->params.sigma_eps - 3.0) < 1e-12);
    assert_true(fabs(m->params.c1_eps - 4.0) < 1e-12);
    assert_true(fabs(m->params.c2_eps - 5.0) < 1e-12);

    turbulence_model_destroy(m);
}

static void test_turbulence_model_destroy_null(void **state)
{
    (void)state;

    // Should be a no-op.
    turbulence_model_destroy(NULL);
}

// 2) Initialization of flow state
static void test_turb_model_initialize_invalid(void **state)
{
    (void)state;
    FlowState fs = {0};

    // Should not crash; arrays remain NULL.
    turb_model_initialize(NULL, NULL, NULL);
    turb_model_initialize(NULL, (Mesh *)1, &fs);
    turb_model_initialize((TurbulenceModel *)1, NULL, &fs);
    turb_model_initialize((TurbulenceModel *)1, (Mesh *)1, NULL);
}

static void test_turb_model_initialize_valid(void **state)
{
    (void)state;
    // Create a 3-node mesh via mesh_create
    double coords[9] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0};
    size_t conn3[3] = {0, 1, 2};
    Mesh *mesh = mesh_create(3, coords, 1, 3, conn3);
    FlowState fs = {0};
    size_t N = mesh_get_num_nodes(mesh);
    fs.turbulence_kinetic_energy = calloc(N, sizeof(double));
    fs.turbulence_dissipation_rate = calloc(N, sizeof(double));

    TurbulenceModel *m = turbulence_model_create(TURBULENCE_MODEL_K_EPSILON, NULL);
    turb_model_initialize(m, mesh, &fs);

    for (size_t i = 0; i < N; ++i)
    {
        assert_float_equal(fs.turbulence_kinetic_energy[i], 1e-3, 1e-12);
        assert_float_equal(fs.turbulence_dissipation_rate[i], 1e-4, 1e-12);
    }

    free(fs.turbulence_kinetic_energy);
    free(fs.turbulence_dissipation_rate);
    turbulence_model_destroy(m);
    mesh_destroy(mesh);
}

// 3) Compute viscosity
static void test_turb_model_compute_viscosity_invalid(void **state)
{
    (void)state;

    // Invalid pointers should return NULL.
    assert_null(turb_model_compute_viscosity(NULL, NULL, NULL));

    Mesh dummy = {.num_nodes = 1};

    FlowState fs = {
        .turbulence_kinetic_energy = (double[]){0},
        .turbulence_dissipation_rate = (double[]){0}};

    assert_null(turb_model_compute_viscosity((TurbulenceModel *)1, &dummy, &fs));
    assert_null(turb_model_compute_viscosity(NULL, &dummy, &fs));
    assert_null(turb_model_compute_viscosity((TurbulenceModel *)1, NULL, &fs));
    assert_null(turb_model_compute_viscosity((TurbulenceModel *)1, &dummy, NULL));
}

static void test_turb_model_compute_viscosity_basic(void **state)
{
    (void)state;

    // Create model with known c_mu
    KEpsilonParameters params = {.c_mu = 0.5, .sigma_k = 0, .sigma_eps = 0, .c1_eps = 0, .c2_eps = 0};
    TurbulenceModel *m = turbulence_model_create(TURBULENCE_MODEL_K_EPSILON, &params);

    Mesh mesh = {.num_nodes = 2};
    FlowState fs = {0};

    fs.turbulence_kinetic_energy = calloc(2, sizeof(double));
    fs.turbulence_dissipation_rate = calloc(2, sizeof(double));
    fs.turbulence_kinetic_energy[0] = 2.0;
    fs.turbulence_dissipation_rate[0] = 0.5;
    fs.turbulence_kinetic_energy[1] = 1e-12;
    fs.turbulence_dissipation_rate[1] = 1e-12;

    double *nu = turb_model_compute_viscosity(m, &mesh, &fs);
    assert_non_null(nu);
    // node0: 0.5 * (2^2)/0.5 = 4.0
    assert_float_equal(nu[0], 4.0, 1e-12);
    // node1: uses floor 1e-10 -> (1e-10)^2/(1e-10) * 0.5 = 0.5e-10
    assert_float_equal(nu[1], 0.5e-10, 1e-20);

    free(nu);
    free(fs.turbulence_kinetic_energy);
    free(fs.turbulence_dissipation_rate);
    turbulence_model_destroy(m);
}

// 4) Update should no-op on invalid and leave state unchanged for trivial mesh
static void test_turb_model_update_invalid(void **state)
{
    (void)state;

    turb_model_update(NULL, NULL, NULL, 0.1);
    turb_model_update((TurbulenceModel *)1, NULL, (FlowState *)1, 0.1);
    turb_model_update(NULL, (Mesh *)1, (FlowState *)1, 0.1);
}

static void test_turb_model_update_noop(void **state)
{
    (void)state;
    // One-node, zero-cell mesh: nb_count < 3, so no modifications
    Mesh mesh = {.num_nodes = 1, .num_cells = 0, .nodes_per_cell = 4, .connectivity = NULL, .coordinates = NULL};
    FlowState fs = {0};

    fs.num_nodes = 1;
    fs.velocity = calloc(3, sizeof(double));
    fs.turbulence_kinetic_energy = calloc(1, sizeof(double));
    fs.turbulence_dissipation_rate = calloc(1, sizeof(double));

    // Seed with non-default.
    fs.turbulence_kinetic_energy[0] = 7.7;
    fs.turbulence_dissipation_rate[0] = 8.8;

    TurbulenceModel *m = turbulence_model_create(TURBULENCE_MODEL_K_EPSILON, NULL);
    turb_model_initialize(m, &mesh, &fs);

    turb_model_update(m, &mesh, &fs, 0.05);

    // Still at least initial values.
    assert_true(fs.turbulence_kinetic_energy[0] >= 1e-10);
    assert_true(fs.turbulence_dissipation_rate[0] >= 1e-10);

    free(fs.velocity);
    free(fs.turbulence_kinetic_energy);
    free(fs.turbulence_dissipation_rate);
    turbulence_model_destroy(m);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_turbulence_model_create_default),
        cmocka_unit_test(test_turbulence_model_create_with_params),
        cmocka_unit_test(test_turbulence_model_destroy_null),
        cmocka_unit_test(test_turb_model_initialize_invalid),
        cmocka_unit_test(test_turb_model_initialize_valid),
        cmocka_unit_test(test_turb_model_compute_viscosity_invalid),
        cmocka_unit_test(test_turb_model_compute_viscosity_basic),
        cmocka_unit_test(test_turb_model_update_invalid),
        cmocka_unit_test(test_turb_model_update_noop),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
