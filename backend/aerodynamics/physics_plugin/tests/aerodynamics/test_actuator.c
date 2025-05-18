#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include "aerodynamics/actuator.h"
#include "aerodynamics/mesh.h"
#include "aerodynamics/flow_state.h"

static void test_actuator_create_destroy(void **state)
{
    (void)state; // unused

    size_t node_ids[] = {0, 1, 2};
    double direction[] = {1.0, 0.0, 0.0};

    Actuator *act = actuator_create("flap", ACTUATOR_TYPE_SURFACE, 3, node_ids, direction, 0.5);
    assert_non_null(act);
    assert_string_equal(act->name, "flap");
    assert_int_equal(act->type, ACTUATOR_TYPE_SURFACE);
    assert_int_equal(act->node_count, 3);
    assert_float_equal(act->gain, 0.5, 1e-8);

    actuator_destroy(act);
}

static void test_actuator_set_command(void **state)
{
    (void)state; // unused

    size_t node_ids[] = {0};
    double direction[] = {0.0, 1.0, 0.0};

    Actuator *act = actuator_create("jet", ACTUATOR_TYPE_BODY_FORCE, 1, node_ids, direction, 1.0);
    assert_non_null(act);

    actuator_set_command(act, 10.0);
    assert_float_equal(act->last_command, 10.0, 1e-8);

    actuator_destroy(act);
}

static void test_actuator_apply_surface(void **state)
{
    (void)state; // unused

    double coords[] = {0, 0, 0,  1, 0, 0,  0, 1, 0};
    size_t conn3[] = {0, 1, 2};
    Mesh *mesh = mesh_create(3, coords, 1, 3, conn3);
    assert_non_null(mesh);

    FlowState fs;
    assert_int_equal(flow_state_init(&fs, mesh), 0);

    size_t node_ids[] = {1, 2};
    double direction[] = {1.0, 0.0, 0.0};
    Actuator *act = actuator_create("aileron", ACTUATOR_TYPE_SURFACE, 2, node_ids, direction, 0.2);
    actuator_set_command(act, 5.0);
    actuator_apply(act, mesh, &fs, 0.01);

    assert_float_equal(fs.velocity[3 * 1], 1.0, 1e-6); // 0.2 * 5.0
    assert_float_equal(fs.velocity[3 * 2], 1.0, 1e-6);
    assert_float_equal(fs.velocity[3 * 0], 0.0, 1e-6); // unaffected node

    actuator_destroy(act);
    flow_state_destroy(&fs);
    mesh_destroy(mesh);
}

static void test_actuator_apply_body_force(void **state)
{
    (void)state; // unused

    double coords[] = {0, 0, 0};
    size_t conn1[] = {0};
    Mesh *mesh = mesh_create(1, coords, 1, 1, conn1);
    assert_non_null(mesh);

    FlowState fs;
    assert_int_equal(flow_state_init(&fs, mesh), 0);

    size_t node_ids[] = {0};
    double direction[] = {0.0, 0.0, 1.0};
    Actuator *act = actuator_create("thruster", ACTUATOR_TYPE_BODY_FORCE, 1, node_ids, direction, 2.0);
    actuator_set_command(act, 3.0);
    actuator_apply(act, mesh, &fs, 0.1);

    assert_float_equal(fs.velocity[2], 0.6, 1e-6); // 2.0 * 3.0 * 0.1 = 0.6

    actuator_destroy(act);
    flow_state_destroy(&fs);
    mesh_destroy(mesh);
}

static void test_actuator_invalid_inputs(void **state)
{
    (void)state; // unused

    Actuator *act = actuator_create(NULL, ACTUATOR_TYPE_SURFACE, 0, NULL, NULL, 0.0);
    assert_null(act);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_actuator_create_destroy),
        cmocka_unit_test(test_actuator_set_command),
        cmocka_unit_test(test_actuator_apply_surface),
        cmocka_unit_test(test_actuator_apply_body_force),
        cmocka_unit_test(test_actuator_invalid_inputs),
    };

    return cmocka_run_group_tests(tests, NULL, NULL);
}
