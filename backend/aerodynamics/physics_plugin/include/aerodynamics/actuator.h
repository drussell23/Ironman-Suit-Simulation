#ifndef AERODYNAMICS_ACTUATOR_H
#define AERODYNAMICS_ACTUATOR_H

#ifdef __cplusplus
extern "C"
{
#endif
#include <stddef.h> // for size_t

    // Forward declarations so parameter types are visible in the API.
    struct Mesh;
    struct FlowState;

    /**
     * Types of actuators supported:
     * - SURFACE: Modifies boundary node velocities (e.g., flaps, control surfaces).
     * - BODY_FORCE: Injects momentum into interior nodes (e.g, thrust jets).
     */
    typedef enum
    {
        ACTUATOR_TYPE_SURFACE = 0,
        ACTUATOR_TYPE_BODY_FORCE
    } ActuatorType;

    /**
     * @brief Represents one actuator that can be commanded each time step.
     */
    typedef struct Actuator
    {
        char *name;
        ActuatorType type; // Type of actuator (e.g., surface, body force).

        /* Geometry mapping: Which nodes this actuator affects */
        size_t node_count; // Number of nodes affected by the actuator.
        size_t *node_ids;  // Array of node IDs affected by the actuator.

        /* Direction & scaling. */
        double direction[3]; // Direction vector of the actuator. < Unit vector (x, y, z) indicating the direction of the actuator.
        double gain;         // Scaling factor for the actuator. < Gain factor to scale the actuator's effect.

        /* Last command value. */
        double last_command; // Last command value sent to the actuator. < Last command value for the actuator
    } Actuator;

    /**
     * @brief Create and configure a new Actuator.
     *
     * @param name null-terminated string name of the actuator. < Name of the actuator.
     * @param type ACTUATOR_TYPE_SURFACE or ACTUATOR_TYPE_BODY_FORCE. < Type of actuator (surface or body force).
     * @param node_count Length of node_ids array. < Number of nodes affected by the actuator.
     * @param node_ids Array of node indices this actuator controls. < Array of node IDs affected by the actuator.
     * @param direction Length-3 array giving actuation direction. (needs to be normalized) < Direction vector of the actuator.
     * @param gain Scaling factor: command -> velocity or force magnitude. < Gain factor to scale the actuator's effect.
     *
     * @return Actuator* Pointer to the created Actuator object, or NULL on failure. < Pointer to the created Actuator object.
     */
    Actuator *actuator_create(const char *name, ActuatorType type, size_t node_count, const size_t *node_ids, const double direction[3], double gain);

    /**
     * @brief Destroy an Actuator, freeing its internal resources.
     * @param act Actuator instance to free. < Pointer to the Actuator object to destroy.
     */
    void actuator_destroy(Actuator *act);

    /**
     * @brief Set the actuator command for this time step.
     *
     * For surface actuators, command is typically an angle (rad) or deflection fraction.
     * For body-force actuators, command is typically a force magnitude (N).
     *
     * @param act Actuator instance to set. < Pointer to the Actuator object.
     * @param command New command value. < New command value for the actuator.
     */
    void actuator_set_command(Actuator *act, double command);

    /**
     * @brief Apply this actuator's effect into the solver.
     *
     * - SURFACE:
     *      Modifies FlowState velocities at boundary nodes:
     *          u_node += gain * command * direction
     * - BODY_FORCE:
     *      Injects a body force proportional to command into FlowState nodal velocities:
     *         u_node += gain * command * dt * direction
     *
     * @param act Actuator instance to apply. < Pointer to the Actuator object.
     * @param mesh Mesh (to validate node indices).
     * @param state Flow state to modify (velocity field).
     * @param dt Current timestep (used for body-force scaling).
     */
    void actuator_apply(const Actuator *act, const struct Mesh *mesh, struct FlowState *state, double dt);

#ifdef __cplusplus
}
#endif // __cplusplus
#endif
