#ifndef AERODYNAMICS_FLOW_STATE_H
#define AERODYNAMICS_FLOW_STATE_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <stddef.h>            // for size_t
#include "aerodynamics/mesh.h" // your Mesh definition

// Forward declare Mesh so flow_state_init() sees it.
struct Mesh; 


    /**
     * @brief Holds all solution fields for one CFD time step.
     *
     * The arrays allocated/owned by the FlowState. Sizes (# of cells or nodes) are obtained from the Mesh
     */
    typedef struct FlowState
    {
        size_t num_nodes; /// < Total number of nodes (mesh->num_nodes)

        double *velocity; /// < Velocity vector (3D) at each node (num_nodes * 3)
        double *pressure;  /// < Pressure at each node (num_nodes)

        /* RANS turbulence fields (only if using k–ε model )*/
        double *turbulence_kinetic_energy;   /// < Turbulence kinetic energy at each node (num_nodes)
        double *turbulence_dissipation_rate; /// < Turbulence dissipation rate at each node (num_nodes)
    } FlowState;

    /**
     * @brief Allocate and zero‐initialize all FlowState arrays.
     *
     * Uses mesh->num_nodes to size every field.  Returns 0 on success,
     * nonzero (e.g. -1) on allocation failure.
     *
     * @param state  Pointer to an uninitialized FlowState.
     * @param mesh   Pointer to a previously initialized Mesh.
     * @return int   0 on success, nonzero on error.
     */
    int flow_state_init(FlowState *state, const struct Mesh *mesh);

    /**
     * @brief Free all arrays inside FlowState.
     *
     * After this call, pointers inside state are invalid.  Does not
     * free the FlowState struct itself.
     *
     * @param state  Pointer to a FlowState previously initialized.
     */
    void flow_state_destroy(FlowState *state);

#ifdef __cplusplus
}
#endif

#endif // AERODYNAMICS_FLOW_STATE_H
