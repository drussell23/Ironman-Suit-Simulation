#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "aerodynamics/actuator.h"
#include "aerodynamics/mesh.h"
#include "aerodynamics/flow_state.h"

/**
 * @brief Create and configure a new Actuator.
 * 
 * @param name null-terminated string name of the actuator.
 * @param type ACTUATOR_TYPE_SURFACE or ACTUATOR_TYPE_BODY_FORCE.
 * @param node_count Length of node_ids array.
 * @param node_ids Array of node indices this actuator controls.
 * @param direction Length-3 array giving actuation direction. (needs to be normalized)
 * @param gain Scaling factor: command -> velocity or force magnitude.
 * 
 * @return Actuator* Pointer to the created Actuator object, or NULL on failure. 
 */
Actuator *actuator_create(const char *name, ActuatorType type, size_t node_count, const size_t *node_ids, const double direction[3], double gain) {
    // Validate input parameters.
    if (!node_ids || node_count == 0) {
        return NULL; // Invalid input
    }

    Actuator *act = (Actuator *)malloc(sizeof(Actuator)); // Allocate memory for Actuator struct.
    
    // Check for allocation failure.
    if (!act) {
        return NULL; // Memory allocation failure
    }

    // Copy name. 
    if (name) {
        size_t len = strlen(name) + 1; // Calculate length of name.
        act->name = (char*)malloc(len); // Allocate memory for name.

        // Check for allocation failure.
        if (!act->name) {
            free(act); // Free previously allocated memory.
            return NULL; // Memory allocation failure
        }

        memcpy(act->name, name, len); // Copy name into Actuator struct.
    } else { 
        act->name = NULL; // Set name to NULL if not provided.
    }

    act->type = type; // Set actuator type.
    act->node_count = node_count; // Set number of nodes.
    act->node_ids = (size_t *)malloc(sizeof(size_t) * node_count); // Allocate memory for node IDs.

    // Check for allocation failure.
    if (!act->node_ids) {
        free(act->name); // Free name if node_ids allocation fails.
        free(act);       // Free Actuator struct.
        return NULL;     // Memory allocation failure
    }

    memcpy(act->node_ids, node_ids, sizeof(size_t) * node_count); // Copy node IDs into Actuator struct.

    // Normalize direction vector.
    double mag = sqrt(direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2]); // Calculate magnitude.

    // Check for zero magnitude.
    if (mag > 1e-12) {
        act->direction[0] = direction[0] / mag; // Normalize x-component.
        act->direction[1] = direction[1] / mag; // Normalize y-component.
        act->direction[2] = direction[2] / mag; // Normalize z-component.
    } else {
        // Default unit x.
        act->direction[0] = 1.0; // Set x-component to 1.
        act->direction[1] = 0.0; // Set y-component to 0.
        act->direction[2] = 0.0; // Set z-component to 0.
    }

    act->gain = gain; // Set gain factor.
    act->last_command = 0.0; // Initialize last command to 0.

    return act; // Return pointer to the created Actuator object.
}

// Set the actuator command for this time step.
void actuator_destroy(Actuator *act) {
    // Free the name if it was allocated.
    if (!act) {
        return; // Nothing to destroy
    }

    free(act->name); // Free name.
    free(act->node_ids); // Free node IDs.
    free(act); // Free Actuator struct.
}

/**
 * @brief Set the actuator command for this time step.
 * 
 * @param act Actuator instance to set.
 * @param command New command value.
 */
void actuator_set_command(Actuator *act, double command) {
    // Set the actuator command for this time step.
    if (!act) {
        return; // Nothing to set
    }

    act->last_command = command; // Store the command value.
}

void actuator_apply(const Actuator *act, const struct Mesh *mesh, struct FlowState *state, double dt) {
    // Check for NULL pointers.
    if (!act || ! mesh || !state) {
        return; // Invalid input
    }

    size_t N = mesh_get_num_nodes(mesh); // Get number of nodes from mesh.

    
    for (size_t idx = 0; idx < act->node_count; ++idx) {
        size_t node = act->node_ids[idx]; // Get node ID from actuator.

        // Check if node ID is valid.
        if (node >= N) {
            continue; // Skip if node ID is out of bounds.
        }

        size_t off = 3 * node; // Calculate offset for 3D velocity vector.
        double cmd = act->last_command; // Get last command value.

        // Apply actuator effect based on type.
        if (act->type == ACTUATOR_TYPE_SURFACE) {
            // Instantaneous velocity change.
            state->velocity[off + 0] += act->gain * cmd * act->direction[0]; // Update x-velocity.
            state->velocity[off + 1] += act->gain * cmd * act->direction[1]; // Update y-velocity.
            state->velocity[off + 2] += act->gain * cmd * act->direction[2]; // Update z-velocity.  
        } else if (act->type == ACTUATOR_TYPE_BODY_FORCE) {
            // Body-force injection: dv = (gain * cmd * dt) * direction
            double factor = act->gain * cmd * dt; // Compute scaling factor.
            state->velocity[off + 0] += factor * act->direction[0]; // Update x-velocity.   
            state->velocity[off + 1] += factor * act->direction[1]; // Update y-velocity.
            state->velocity[off + 2] += factor * act->direction[2]; // Update z-velocity.
        }
    }
}