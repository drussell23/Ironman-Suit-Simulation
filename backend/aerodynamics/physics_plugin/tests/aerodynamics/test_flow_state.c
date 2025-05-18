// tests/aerodynamic/test_flow_state.c

/*
Enable the allocation-failure test on Linux (where GNU Id supports --wrap=calloc).
Disable that test on platforms (macOS, Windows) where the wrap mechanism isn't available, so your test suite still builds and runs everywhere.
*/
#if defined(__linux__) && !defined(__APPLE__) && !defined(_WIN32)
#define MOCK_CALLOC
#endif

#include <stdio.h>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cmocka.h>

#include "aerodynamics/flow_state.h"
#include "aerodynamics/mesh.h"

// --- A minimal fake Mesh implementation for testing ---
static Mesh *mesh_create_fake(size_t num_nodes)
{
    Mesh *m = malloc(sizeof(Mesh));

    if (!m)
    {
        return NULL;
    }

    m->num_nodes = num_nodes;
    m->nodes_per_cell = 1;
    m->num_cells = 0;
    m->coordinates = NULL;
    m->connectivity = NULL;

    return m;
}

static void mesh_destroy_fake(Mesh *m)
{
    free(m);
}

// Override the real mesh_get_num_nodes to point at our fake:
size_t mesh_get_num_nodes(const Mesh *m)
{
    return m->num_nodes;
}

// --- Colloc wrappers for testing allocation failures ----------------------

#ifdef MOCK_CALLOC
// CMocka’s wrap mechanism: we must declare the wrapped symbol
void *__wrap_calloc(size_t nmemb, size_t size);
#endif

// -- Test: successful init + destroy --

static void test_flow_state_init_and_destroy(void **state)
{
    (void)state;

    Mesh *fake = mesh_create_fake(42);
    assert_non_null(fake);

    FlowState fs;
    memset(&fs, 0xAA, sizeof(fs));

    // init should succeed.
    int rc = flow_state_init(&fs, fake);
    assert_int_equal(rc, 0);

    // Check num_nodes was set.
    assert_int_equal(fs.num_nodes, 42);

    // All pointers non-NULL.
    assert_non_null(fs.velocity);
    assert_non_null(fs.pressure);
    assert_non_null(fs.turbulence_kinetic_energy);
    assert_non_null(fs.turbulence_dissipation_rate);

    // Each allocation should be the correct size.
    // velocity: 3 * nodes doubles
    for (size_t i = 0; i < fs.num_nodes * 3; ++i)
    {
        // calloc zeros the memory.
        assert_float_equal(fs.velocity[i], 0.0, 0.0);
    }

    for (size_t i = 0; i < fs.num_nodes; ++i)
    {
        assert_float_equal(fs.pressure[i], 0.0, 0.0);
        assert_float_equal(fs.turbulence_kinetic_energy[i], 0.0, 0.0);
        assert_float_equal(fs.turbulence_dissipation_rate[i], 0.0, 0.0);
    }

    // Destroy should free and reset fields.
    flow_state_destroy(&fs);
    assert_null(fs.velocity);
    assert_null(fs.pressure);
    assert_null(fs.turbulence_kinetic_energy);
    assert_null(fs.turbulence_dissipation_rate);
    assert_int_equal(fs.num_nodes, 0);

    mesh_destroy_fake(fake);
}

// -- Test: init with zero nodes should still succeed but allocs size zero --

static void test_flow_state_init_zero_nodes(void **state)
{
    (void)state;

    Mesh *fake = mesh_create_fake(0);
    assert_non_null(fake);

    FlowState fs;
    memset(&fs, 0, sizeof(fs));

    int rc = flow_state_init(&fs, fake);
    assert_int_equal(rc, 0);

    // Even with zero nodes, pointers come back as non-NULL (calloc(0) may)
    assert_non_null(fs.velocity);
    assert_non_null(fs.pressure);

    // But no element to read.
    assert_int_equal(fs.num_nodes, 0);

    flow_state_destroy(&fs);
    mesh_destroy_fake(fake);
}

// -- Test: double init/destroy should be safe --

static void test_flow_state_double_init_destroy(void **state)
{
    (void)state;

    Mesh *fake = mesh_create_fake(5);
    assert_non_null(fake);

    FlowState fs = {0};

    for (int cycle = 0; cycle < 2; ++cycle)
    {
        assert_int_equal(flow_state_init(&fs, fake), 0);
        assert_int_equal(fs.num_nodes, 5);

        flow_state_destroy(&fs);

        assert_int_equal(fs.num_nodes, 0);
        assert_null(fs.velocity);
    }

    mesh_destroy_fake(fake);
}

// -- Test fixture for simulating allocation failure --

static int fake_alloc_fail_setup(void **state)
{
    (void)state;
#ifdef MOCK_CALLOC
    // Enqueue a NULL return for the first calloc call
    will_return(__wrap_calloc, NULL);
#endif
    return 0;
}

static int fake_calloc_teardown(void **state)
{
    (void)state;
#ifdef MOCK_CALLOC
    // Consume the queued NULL so later tests are unaffected
    (void)__wrap_calloc(1, 1);
#endif
    return 0;
}
static void test_flow_state_init_alloc_fail(void **state)
{
    (void)state;

    // Create a small mesh
    Mesh *fake = mesh_create_fake(3);
    assert_non_null(fake);

    FlowState fs;
    memset(&fs, 0, sizeof(fs));

    // We’ll wrap calloc to return NULL on first invocation
    expect_any(__wrap_calloc, nmemb);
    expect_any(__wrap_calloc, size);
    will_return(__wrap_calloc, NULL);

    int rc = flow_state_init(&fs, fake);

    // should detect allocation failure and return -1
    assert_int_equal(rc, -1);

    // All pointers should be NULL/reset
    assert_null(fs.velocity);
    assert_null(fs.pressure);
    assert_int_equal(fs.num_nodes, 0);

    mesh_destroy_fake(fake);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_flow_state_init_and_destroy),
        cmocka_unit_test(test_flow_state_init_zero_nodes),
        cmocka_unit_test(test_flow_state_double_init_destroy),
#ifdef MOCK_CALLOC
        cmocka_unit_test_setup_teardown(
            test_flow_state_init_alloc_fail,
            fake_alloc_fail_setup,
            fake_calloc_teardown),
#endif
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
