// tests/aerodynamics/test_mesh.c

#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "aerodynamics/mesh.h"
#include "aerodynamics/flow_state.h"

//
// Test mesh_create rejects bad input and succeeds on valid input.
// (mesh_create from mesh.c :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1})
//
static void test_mesh_create_invalid(void **state)
{
    (void)state;

    double coords[3] = {0.0, 0.0, 0.0};
    size_t conn1[1] = {0};

    assert_null(mesh_create(0, coords, 1, 1, conn1));
    assert_null(mesh_create(1, NULL, 1, 1, conn1));
}

static void test_mesh_create_valid_and_destroy(void **state)
{
    (void)state;

    const size_t N = 2;
    double coords[6] = {0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    size_t conn2[2] = {0, 1};
    Mesh *m = mesh_create(N, coords, 1, 2, conn2);

    assert_non_null(m);
    assert_int_equal(m->num_nodes, N);

    // Coordinates should have been copied.
    for (size_t i = 0; i < N * 3; ++i)
    {
        assert_true(fabs(m->coordinates[i] - coords[i]) < 1e-12);
    }

    // Destruction should safely free.
    mesh_destroy(m);
}

static void test_mesh_destroy_null(void **state)
{
    (void)state;

    // Should be no-op.
    mesh_destroy(NULL);
}

//
// mesh_get_num_nodes should return the stored value or 0 on NULL.
// (mesh_get_num_nodes from mesh.c :contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3})
//
static void test_mesh_get_num_nodes(void **state)
{
    (void)state;

    Mesh tmp = {.num_nodes = 5};

    assert_int_equal(mesh_get_num_nodes(&tmp), 5);
    assert_int_equal(mesh_get_num_nodes(NULL), 0);
}

//
// mesh_initialize should reject NULL, handle empty meshes, and compute
// volumes + adjacency correctly for a single tetra.
// (mesh_initialize + share_face logic in mesh.c :contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5})
//
static void test_mesh_initialize_null(void **state)
{
    (void)state;

    assert_int_equal(mesh_initialize(NULL), -1);
}

static void test_mesh_initialize_empty(void **state)
{
    (void)state;
    Mesh m = {0};

    m.num_cells = 0;
    m.nodes_per_cell = 4;

    // Even with no nodes/cells, init should succeed.
    assert_int_equal(mesh_initialize(&m), 0);
    assert_non_null(m.cell_volumes);
    assert_non_null(m.cell_adjacency);
    assert_non_null(m.adjacency_offsets);

    // Only one offset entry == 0
    assert_int_equal(m.adjacency_offsets[0], 0);

    // Cleanup.
    free(m.cell_volumes);
    free(m.cell_adjacency);
    free(m.adjacency_offsets);
}

static void test_mesh_initialize_tetra(void **state)
{
    (void)state;

    // Build a unit tetra: nodes at (0,0,0),(1,0,0),(0,1,0),(0,0,1)
    Mesh m = {0};

    m.num_nodes = 4;
    m.coordinates = malloc(sizeof(double) * 12);

    double coords[12] = {
        0, 0, 0,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1};

    memcpy(m.coordinates, coords, sizeof(coords));

    m.nodes_per_cell = 4;
    m.num_cells = 1;
    m.connectivity = malloc(sizeof(size_t) * 4);

    m.connectivity[0] = 0;
    m.connectivity[1] = 1;
    m.connectivity[2] = 2;
    m.connectivity[3] = 3;

    assert_int_equal(mesh_initialize(&m), 0);

    // Volume of unit tetra is 1/6.
    assert_float_equal(m.cell_volumes[0], 1.0 / 6.0, 1e-12);

    // No neighbors => both offsets zero.
    assert_int_equal(m.adjacency_offsets[0], 0);
    assert_int_equal(m.adjacency_offsets[1], 0);

    // Cleanup.
    free(m.coordinates);
    free(m.connectivity);
    free(m.cell_volumes);
    free(m.cell_adjacency);
    free(m.adjacency_offsets);
}

//
// mesh_compute_diffusion should apply
//   v_new = v_old - (nu*dt) * v_old = v_old*(1 - nu*dt)
// (mesh_compute_diffusion in mesh.c :contentReference[oaicite:6]{index=6}:contentReference[oaicite:7]{index=7})
//
static void test_mesh_compute_diffusion(void **state)
{
    (void)state;

    // Fake mesh and flow state.
    Mesh m = {.num_nodes = 2};
    FlowState fs = {.num_nodes = 2};

    fs.velocity = calloc(3 * 2, sizeof(double));

    // Initialize velocities: node0=(1,2,3), node1=(4,5,6)
    fs.velocity[0] = 1;
    fs.velocity[1] = 2;
    fs.velocity[2] = 3;
    fs.velocity[3] = 4;
    fs.velocity[4] = 5;
    fs.velocity[5] = 6;

    double nu_t[2] = {0.5, 1.0};
    double dt = 0.1;

    mesh_compute_diffusion(&m, &fs, nu_t, dt);

    // node0: factor=0.05 => v0 = old*(1 - 0.05)
    assert_float_equal(fs.velocity[0], 1.0 * (1 - 0.5 * dt), 1e-12);
    assert_float_equal(fs.velocity[1], 2.0 * (1 - 0.5 * dt), 1e-12);
    assert_float_equal(fs.velocity[2], 3.0 * (1 - 0.5 * dt), 1e-12);

    // node1: factor=0.1 => v1 = old*(1 - 0.1)
    assert_float_equal(fs.velocity[3], 4.0 * (1 - 1.0 * dt), 1e-12);
    assert_float_equal(fs.velocity[4], 5.0 * (1 - 1.0 * dt), 1e-12);
    assert_float_equal(fs.velocity[5], 6.0 * (1 - 1.0 * dt), 1e-12);

    free(fs.velocity);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_mesh_create_invalid),
        cmocka_unit_test(test_mesh_create_valid_and_destroy),
        cmocka_unit_test(test_mesh_destroy_null),
        cmocka_unit_test(test_mesh_get_num_nodes),
        cmocka_unit_test(test_mesh_initialize_null),
        cmocka_unit_test(test_mesh_initialize_empty),
        cmocka_unit_test(test_mesh_initialize_tetra),
        cmocka_unit_test(test_mesh_compute_diffusion),
    };

    return cmocka_run_group_tests(tests, NULL, NULL);
}
