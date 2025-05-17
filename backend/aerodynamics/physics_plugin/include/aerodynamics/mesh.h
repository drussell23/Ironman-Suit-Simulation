#ifndef AERODYNAMICS_MESH_H
#define AERODYNAMICS_MESH_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <math.h> // for sqrt
#include <string.h>
#include <stdlib.h> // for malloc, free
#include <stddef.h> // for size_t

    typedef struct FlowState FlowState;

    /**
     * @brief Represents the computational mesh: node count & coordinates.
     */
    typedef struct Mesh
    {
        // User-provided geometry:
        size_t num_nodes;    /// < Total number of nodes in the mesh.
        double *coordinates; /// < Array of node coordinates (num_nodes * 3).

        // User-provided topology:
        size_t num_cells;      /// < Total number of cells in the mesh.
        size_t nodes_per_cell; /// < Number of nodes per cell (e.g., 4 for tetrahedral mesh) (e.g. 4 for tets, 8 for hexahedra).
        size_t *connectivity;  /// < Array of node indices for each cell (num_cells * nodes_per_cell).

        // Computed by mesh_initialize():
        double *cell_volumes;      /// < Array of cell volumes (num_cells).
        size_t *adjacency_offsets; /// < Array of offsets for adjacency list (num_cells + 1).
        size_t *cell_adjacency;    /// < Flattened adjacency list (num_cells * max_adj_cells).
    } Mesh;

    /**
     * @brief Create a Mesh object by copying user-provided coordinates.
     *
     * @param num_nodes Number of nodes in the mesh.
     * @param coords  Pointer to an array of node coordinates (num_nodes * 3).
     * @return Mesh* Pointer to the created Mesh object, or NULL on failure.
     */
    Mesh *mesh_create(size_t num_nodes, const double *coords);

    /**
     * @brief Destroy the mesh, freeing its memory.
     * @param mesh Pointer to the Mesh object to destroy.
     */
    void mesh_destroy(Mesh *mesh);

    /**
     * @brief Initialize the mesh by computing connectivity, cell volumes, etc.
     *
     * @param mesh Pointer to the Mesh object to initialize.
     * @return int 0 on success, nonzero on error.
     */
    int mesh_initialize(Mesh *mesh);

    /**
     * @brief Query number of nodes.
     * @param mesh Pointer to the Mesh object.
     * @return size_t Number of nodes in the mesh.
     */
    size_t mesh_get_num_nodes(const Mesh *mesh);

    /**
     * @brief Compute convection term. Placeholder for actual implementation.
     * @param mesh Pointer to the Mesh object.
     * @param state Pointer to the FlowState object.
     * @param dt Time step size.
     */
    void mesh_compute_convection(const Mesh *mesh, FlowState *state, double dt);

    /**
     * @brief Compute diffusion term using eddy viscosity.
     * Applies a simple decay: v_i ← v_i − (ν_t[i]·dt)·v_i
     *
     * @param mesh Pointer to the Mesh object.
     * @param state FlowState to update (velocity field).
     * @param nu_t Array of length num_nodes containing eddy viscosity values.
     * @param dt Time step size.
     */
    void mesh_compute_diffusion(const Mesh *mesh, FlowState *state, const double *nu_t, double dt);

#ifdef __cplusplus
}
#endif

#endif // AERODYNAMICS_MESH_H
