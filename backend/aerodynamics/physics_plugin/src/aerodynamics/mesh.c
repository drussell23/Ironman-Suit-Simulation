#include <stdlib.h> // for malloc, free
#include <string.h> // for memset

#include "aerodynamics/mesh.h"       // For Mesh
#include "aerodynamics/flow_state.h" // For FlowState

// Create a new Mesh object.
Mesh *mesh_create(size_t num_nodes, const double *coords, size_t num_cells, size_t nodes_per_cell, const size_t *connectivity)
{
    if (num_nodes == 0 || !coords || num_cells == 0 || nodes_per_cell == 0 || !connectivity) // Check for invalid input.
    {
        return NULL; // Invalid input
    }

    Mesh *m = malloc(sizeof(*m));

    if (!m)
    {
        return NULL;
    }

    m->num_nodes = num_nodes;
    m->coordinates = malloc(sizeof(double) * 3 * num_nodes);
    m->num_cells = num_cells;
    m->nodes_per_cell = nodes_per_cell;
    m->connectivity = malloc(sizeof(size_t) * num_cells * nodes_per_cell);

    if (!m->coordinates || !m->connectivity)
    {
        free(m->coordinates);
        free(m->connectivity);
        free(m);

        return NULL;
    }

    memcpy(m->coordinates, coords, sizeof(double) * 3 * num_nodes);
    memcpy(m->connectivity, connectivity, sizeof(size_t) * num_cells * nodes_per_cell);

    return m;
}

// Free the Mesh object.
void mesh_destroy(Mesh *mesh)
{
    if (!mesh)
        return;
    free(mesh->coordinates);
    free(mesh->connectivity); // ← free the connectivity array you allocated
    free(mesh);
}

// This function checks if two cells share a face. It compares the connectivity arrays of two cells and counts the number of common nodes.
static int share_face(const size_t *connA, const size_t *connB, size_t nodes_per_cell)
{
    size_t common = 0; // Initialize common node count.

    for (size_t i = 0; i < nodes_per_cell; ++i)
    { // Iterate through nodes of the first cell.
        for (size_t j = 0; j < nodes_per_cell; ++j)
        { // Iterate through nodes of the second cell.
            if (connA[i] == connB[j])
            {             // Check if nodes are the same.
                ++common; // Increment common node count.
                break;    // Break inner loop if a match is found.
            }
        }
    }
    // Check if they share a face. For a face in a tethrahedral mesh, 3 nodes must be common.
    return (common == nodes_per_cell - 1); // Return true if they share a face.
}

int mesh_initialize(Mesh *mesh)
{
    if (!mesh) // Check for NULL pointer.
    {
        return -1; // Invalid input
    }

    size_t C = mesh->num_cells;      // Number of cells.
    size_t P = mesh->nodes_per_cell; // Number of nodes per cell.

    // 1) Allocate memory for cell volumes.
    mesh->cell_volumes = (double *)malloc(sizeof(double) * C);

    if (!mesh->cell_volumes) // Check for allocation failure.
    {
        return -1; // Memory allocation failure
    }

    // 2) Compute volumes (asusmes simple tetra: P == 4).
    for (size_t c = 0; c < C; ++c)
    {                                                    // Iterate through cells.
        const size_t *conn = mesh->connectivity + c * P; // Get connectivity for cell c.

        // Load coordinates of vertices.
        double x0 = mesh->coordinates[3 * conn[0] + 0]; // X coordinate of vertex 0.
        double y0 = mesh->coordinates[3 * conn[0] + 1]; // Y coordinate of vertex 0.
        double z0 = mesh->coordinates[3 * conn[0] + 2]; // Z coordinate of vertex 0.
        double v1[3], v2[3], v3[3];                     // Vectors for other vertices.

        for (size_t k = 0; k < 3; ++k)
        {                                                                                        // Iterate through dimensions.
            v1[k] = mesh->coordinates[3 * conn[1] + k] - (k == 0 ? x0 : (k == 1 ? y0 : z0)),     // Vector from vertex 0 to vertex 1.
                v2[k] = mesh->coordinates[3 * conn[2] + k] - (k == 0 ? x0 : (k == 1 ? y0 : z0)), // Vector from vertex 0 to vertex 2.
                v3[k] = mesh->coordinates[3 * conn[3] + k] - (k == 0 ? x0 : (k == 1 ? y0 : z0)); // Vector from vertex 0 to vertex 3.
        }

        // Cross product of v1 and v2.
        double cx = v1[1] * v2[2] - v1[2] * v2[1]; // X component of cross product.
        double cy = v1[2] * v2[0] - v1[0] * v2[2]; // Y component of cross product.
        double cz = v1[0] * v2[1] - v1[1] * v2[0]; // Z component of cross product.

        // Dot product with v3.
        double dot = cx * v3[0] + cy * v3[1] + cz * v3[2]; // Dot product.

        mesh->cell_volumes[c] = fabs(dot) / 6.0; // Volume of tetrahedron.
    }

    // 3) Build adjacency lists. Over-allocate: each cell can have at most P faces.
    size_t max_adj = C * P; // Maximum adjacency size.

    mesh->cell_adjacency = (size_t *)malloc(sizeof(size_t) * max_adj);    // Allocate memory for adjacency list.
    mesh->adjacency_offsets = (size_t *)malloc(sizeof(size_t) * (C + 1)); // Allocate memory for offsets.

    // Check for allocation failure.
    if (!mesh->cell_adjacency || !mesh->adjacency_offsets)
    {
        free(mesh->cell_volumes); // Free previously allocated memory.
        return -1;                // Memory allocation failure.
    }

    mesh->adjacency_offsets[0] = 0; // Initialize first offset.

    for (size_t i = 0; i < C; ++i) // Iterate through cells.
    {
        size_t count = 0;                                  // Initialize count of adjacent cells.
        const size_t *conn_i = mesh->connectivity + i * P; // Get connectivity for cell i.

        // Iterate through cells for adjacency check.
        for (size_t j = 0; j < C; ++j)
        {
            if (i == j) // Check if comparing the same cell.
            {
                continue; // Skip self-comparison.
            }

            const size_t *conn_j = mesh->connectivity + j * P; // Get connectivity for cell j.

            if (share_face(conn_i, conn_j, P)) // Check if cells share a face.
            {
                mesh->cell_adjacency[i * P + count] = j; // Add cell j to adjacency list of cell i.
                ++count;                                 // Increment count of adjacent cells.
            }
        }
        mesh->adjacency_offsets[i + 1] = mesh->adjacency_offsets[i] + count; // Set offset for next cell.
    }

    return 0; // Success.
}

size_t mesh_get_num_nodes(const Mesh *mesh)
{
    return mesh ? mesh->num_nodes : 0; // Return number of nodes, or 0 if mesh is NULL.
}

/**
 * @brief Compute convection on an unstructured mesh via nodal least-squares gradient.
 *
 * ∂u/∂t + (u·∇)u = 0  ⇒  u_new = u_old + dt * [ -(u·∇)u ].
 *
 * We approximate ∇u at each node by solving a small 3×3 least‐squares system:
 *   ∑_j (r_ij ⊗ r_ij) · ∇u = ∑_j (r_ij ⊗ (u_j − u_i))
 * where r_ij = x_j − x_i.
 *
 * @param mesh   Mesh struct (must contain connectivity, coords, num_cells, nodes_per_cell)
 * @param state  FlowState with nodal velocity (flattened [i*3 + {0,1,2}])
 * @param dt     Timestep size
 */
/**
 * @brief Compute convection on an unstructured mesh via nodal least-squares gradient.
 *
 * ∂u/∂t + (u·∇)u = 0  ⇒  u_new = u_old + dt * [ -(u·∇)u ].
 *
 * We approximate ∇u at each node by solving a small 3×3 least‐squares system:
 *   ∑_j (r_ij ⊗ r_ij) · ∇u = ∑_j (r_ij ⊗ (u_j − u_i))
 * where r_ij = x_j − x_i.
 *
 * @param mesh   Mesh struct (must contain connectivity, coords, num_cells, nodes_per_cell)
 * @param state  FlowState with nodal velocity (flattened [i*3 + {0,1,2}])
 * @param dt     Timestep size
 */
void mesh_compute_convection(
    const Mesh *mesh,
    FlowState *state,
    double dt)
{
    if (!mesh || !state || dt <= 0.0) // Check for NULL pointers and non-positive dt.
        return;                       // Invalid input.

    const size_t N = mesh->num_nodes;      // Get number of nodes.
    const size_t C = mesh->num_cells;      // Get number of cells.
    const size_t P = mesh->nodes_per_cell; // Get number of nodes per cell.

    const double *coords = mesh->coordinates; // Get coordinates.

    double *vel_new = state->velocity; // Get velocity array.

    // 1) Copy old velocity.
    double *vel_old = malloc(sizeof(double) * 3 * N); // Allocate memory for old velocity.

    // Check for allocation failure.
    if (!vel_old)
    {
        return; // Memory allocation failure
    }

    // Copy old velocity into vel_old.
    memcpy(vel_old, state->velocity, sizeof(double) * 3 * N);

    // Buffers for per-node convective acceleration.
    double *accel = calloc(3 * N, sizeof(double));

    // Check for allocation failure.
    if (!accel)
    {
        free(vel_old); // Free previously allocated memory.
        return;        // Memory allocation failure.
    }

    // Temporary arrays for neighbor collection.
    size_t *neighbors = malloc(sizeof(size_t) * N); // Allocate memory for neighbors.
    uint8_t *visited = calloc(N, 1);                // Allocate memory for visited nodes.

    // Check for allocation failure.
    if (!neighbors || !visited)
    {
        free(vel_old);   // Free previously allocated memory.
        free(accel);     // Free previously allocated memory.
        free(neighbors); // Free previously allocated memory.
        free(visited);   // Free previously allocated memory.

        return; // Memory allocation failure.
    }

    // For each node, build least-squares system:
    for (size_t i = 0; i < N; ++i) // Iterate through the nodes.
    {
        // 2) Gather unique neighbors of node i via connectivity
        memset(visited, 0, N); // Reset visited array.
        size_t nb_count = 0;   // Initialize neighbor count.

        for (size_t c = 0; c < C; ++c) // Iterate through cells.
        {
            const size_t *conn = mesh->connectivity + c * P; // Get connectivity for cell c.

            // Check membership.
            int in_cell = 0; // Initialize in_cell flag.

            for (size_t k = 0; k < P; ++k) // Iterate through nodes in the cell.
            {
                if (conn[k] == i) // Check if node i is in the cell.
                {
                    in_cell = 1; // Set in_cell flag.
                    break;       // Break if found.
                }
            }

            if (!in_cell) // Check if node i is not in the cell.
            {
                continue; // Skip if node i is not in the cell.
            }

            // Add all other nodes from this cell.
            for (size_t k = 0; k < P; ++k) // Iterate through nodes in the cell.
            {
                size_t j = conn[k]; // Get node index.

                if (j == i || visited[j]) // Check if node is itself or already visited.
                {
                    continue; // Skip if node is itself or already visited.
                }

                visited[j] = 1;            // Mark node as visited.
                neighbors[nb_count++] = j; // Add node to neighbors list.
            }
        }

        // Skip if too few neighbors to form a 3 × 3 system.
        if (nb_count < 3) // Check if neighbor count is less than 3.
        {
            accel[3 * i + 0] = accel[3 * i + 1] = accel[3 * i + 2] = 0.0; // Set acceleration to zero.
            continue;                                                     // Skip to next node.
        }

        // 3) Assemble AtA and Atb for least-squares: AtA ∈ R^{3×3}, Atb ∈ R^{3×3}
        double AtA[3][3] = {{0}}, Atb[3][3] = {{0}}; // Initialize matrices.

        // Node i’s coordinates & velocity.
        double xi = coords[3 * i + 0], // X coordinate of node i.
            yi = coords[3 * i + 1],    // Y coordinate of node i.
            zi = coords[3 * i + 2];    // Z coordinate of node i.

        // Node i’s velocity.
        double ui[3] = {                     // Velocity of node i.
                        vel_old[3 * i + 0],  // X velocity of node i.
                        vel_old[3 * i + 1],  // Y velocity of node i.
                        vel_old[3 * i + 2]}; // Z velocity of node i.

        // Iterate through neighbors.
        for (size_t n = 0; n < nb_count; ++n)
        {
            size_t j = neighbors[n]; // Get neighbor index.

            // r_ij = x_j - x_i
            double rx = coords[3 * j + 0] - xi; // X component of r_ij.
            double ry = coords[3 * j + 1] - yi; // Y component of r_ij.
            double rz = coords[3 * j + 2] - zi; // Z component of r_ij.

            // Δu = u_j - u_i
            double dux = vel_old[3 * j + 0] - ui[0]; // X component of velocity difference.
            double duy = vel_old[3 * j + 1] - ui[1]; // Y component of velocity difference.
            double duz = vel_old[3 * j + 2] - ui[2]; // Z component of velocity difference.

            // Accumulate AtA[k][l] += r[k]*r[l]
            AtA[0][0] += rx * rx; // X component of AtA.
            AtA[0][1] += rx * ry; // Y component of AtA.
            AtA[0][2] += rx * rz; // Z component of AtA.
            AtA[1][0] += ry * rx; // X component of AtA.
            AtA[1][1] += ry * ry; // Y component of AtA.
            AtA[1][2] += ry * rz; // Z component of AtA.
            AtA[2][0] += rz * rx; // X component of AtA.
            AtA[2][1] += rz * ry; // Y component of AtA.
            AtA[2][2] += rz * rz; // Z component of AtA.

            // accumulate Atb[k][l] += r[k]*du[l]
            Atb[0][0] += rx * dux; // X component of Atb.
            Atb[0][1] += rx * duy; // Y component of Atb.
            Atb[0][2] += rx * duz; // Z component of Atb.
            Atb[1][0] += ry * dux; // X component of Atb.
            Atb[1][1] += ry * duy; // Y component of Atb.
            Atb[1][2] += ry * duz; // Z component of Atb.
            Atb[2][0] += rz * dux; // X component of Atb.
            Atb[2][1] += rz * duy; // Y component of Atb.
            Atb[2][2] += rz * duz; // Z component of Atb.
        }

        // 4) Solve 3×3 linear system: ∇u = inv(AtA) * Atb

        // Compute determinant of AtA.
        double det =
            AtA[0][0] * (AtA[1][1] * AtA[2][2] - AtA[1][2] * AtA[2][1]) - AtA[0][1] * (AtA[1][0] * AtA[2][2] - AtA[1][2] * AtA[2][0]) + AtA[0][2] * (AtA[1][0] * AtA[2][1] - AtA[1][1] * AtA[2][0]);

        // Check for singular matrix (det = 0).
        // If the determinant is close to zero, set acceleration to zero.
        // This indicates that the system is singular or nearly singular.
        // In such cases, we set the acceleration to zero to avoid numerical instability.
        if (fabs(det) < 1e-12)
        {
            accel[3 * i + 0] = accel[3 * i + 1] = accel[3 * i + 2] = 0.0; // Set acceleration to zero.
            continue;                                                     // Skip to next node.
        }

        // Compute inverse of AtA using Cramer's rule.
        // invA[l][m] = 1/det * (AtA[l][l] * AtA[m][m] - AtA[l][m] * AtA[m][l])
        // where l, m = 0, 1, 2.
        double invA[3][3];
        invA[0][0] = (AtA[1][1] * AtA[2][2] - AtA[1][2] * AtA[2][1]) / det;  // X component of invA.
        invA[0][1] = -(AtA[0][1] * AtA[2][2] - AtA[0][2] * AtA[2][1]) / det; // Y component of invA.
        invA[0][2] = (AtA[0][1] * AtA[1][2] - AtA[0][2] * AtA[1][1]) / det;  // Z component of invA.
        invA[1][0] = -(AtA[1][0] * AtA[2][2] - AtA[1][2] * AtA[2][0]) / det; // X component of invA.
        invA[1][1] = (AtA[0][0] * AtA[2][2] - AtA[0][2] * AtA[2][0]) / det;  // Y component of invA.
        invA[1][2] = -(AtA[0][0] * AtA[1][2] - AtA[0][2] * AtA[1][0]) / det; // Z component of invA.
        invA[2][0] = (AtA[1][0] * AtA[2][1] - AtA[1][1] * AtA[2][0]) / det;  // X component of invA.
        invA[2][1] = -(AtA[0][0] * AtA[2][1] - AtA[0][1] * AtA[2][0]) / det; // Y component of invA.
        invA[2][2] = (AtA[0][0] * AtA[1][1] - AtA[0][1] * AtA[1][0]) / det;  // Z component of invA.

        // Compute gradient: grad[l][m] = ∑_k invA[l][k] * Atb[k][m]
        double grad[3][3] = {{0}}; // Initialize gradient array.

        for (size_t l = 0; l < 3; ++l) // Iterate through gradient dimensions.
        {
            for (size_t m = 0; m < 3; ++m) // Iterate through gradient dimensions.
            {
                double sum = 0.0; // Initialize sum for gradient component.

                for (size_t k = 0; k < 3; ++k) // Iterate through AtA dimensions.
                {
                    sum += invA[l][k] * Atb[k][m]; // Accumulate gradient component.
                }
                grad[l][m] = sum; // Store computed gradient component.
            }
        }

        // 5) Convective acceleration a = -(u·∇)u

        // Compute convective acceleration using the gradient.
        // a = -(u·∇)u = -[u[0] * grad[0][0] + u[1] * grad[1][0] + u[2] * grad[2][0],
        //                u[0] * grad[0][1] + u[1] * grad[1][1] + u[2] * grad[2][1],
        //                u[0] * grad[0][2] + u[1] * grad[1][2] + u[2] * grad[2][2]]
        // where u = [u[0], u[1], u[2]] is the velocity vector at node i.
        // The acceleration is stored in the accel array.
        // The acceleration is negative because we are solving the equation in the form ∂u/∂t + (u·∇)u = 0.
        // The negative sign indicates that the acceleration is in the opposite direction of the velocity gradient.
        accel[3 * i + 0] = -(ui[0] * grad[0][0] + ui[1] * grad[1][0] + ui[2] * grad[2][0]); // X component of acceleration.
        accel[3 * i + 1] = -(ui[0] * grad[0][1] + ui[1] * grad[1][1] + ui[2] * grad[2][1]); // Y component of acceleration.
        accel[3 * i + 2] = -(ui[0] * grad[0][2] + ui[1] * grad[1][2] + ui[2] * grad[2][2]); // Z component of acceleration.
    }

    // 6) Update velocities.

    // Apply the computed acceleration to the old velocity to get the new velocity.
    for (size_t i = 0; i < N; ++i)
    {
        vel_new[3 * i + 0] = vel_old[3 * i + 0] + dt * accel[3 * i + 0]; // Update x-velocity.
        vel_new[3 * i + 1] = vel_old[3 * i + 1] + dt * accel[3 * i + 1]; // Update y-velocity.
        vel_new[3 * i + 2] = vel_old[3 * i + 2] + dt * accel[3 * i + 2]; // Update z-velocity.
    }

    free(vel_old);   // Free old velocity array.
    free(accel);     // Free acceleration array.
    free(neighbors); // Free neighbors array.
    free(visited);   // Free visited array.
    // Note: The memory for vel_new is not freed here because it is part of the FlowState struct.
    // The caller is responsible for freeing the FlowState struct when it is no longer needed.
    // The memory for the FlowState struct should be freed using flow_state_destroy().
}

void mesh_compute_diffusion(const Mesh *mesh, FlowState *state, const double *nu_t, double dt)
{
    // Check for NULL pointers.
    if (!mesh || !state || !nu_t)
    {
        return; // Invalid input
    }

    size_t n = mesh->num_nodes; // Get number of nodes from mesh.

    for (size_t i = 0; i < n; ++i)
    {
        size_t off = i * 3;           // Calculate offset for 3D velocity vector.
        double factor = nu_t[i] * dt; // Compute diffusion factor.

        state->velocity[off + 0] -= factor * state->velocity[off + 0]; // Update x-velocity.
        state->velocity[off + 1] -= factor * state->velocity[off + 1]; // Update y-velocity.
        state->velocity[off + 2] -= factor * state->velocity[off + 2]; // Update z-velocity.
    }
}
