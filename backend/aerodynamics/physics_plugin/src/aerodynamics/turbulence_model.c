#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "aerodynamics/turbulence_model.h"
#include "aerodynamics/mesh.h"
#include "aerodynamics/flow_state.h"

TurbulenceModel *turbulence_model_create(TurbulenceModelType type, const KEpsilonParameters *params)
{
    TurbulenceModel *model = (TurbulenceModel *)malloc(sizeof(TurbulenceModel)); // Allocate memory for the model.

    // Check for allocation failure.
    if (!model)
    {
        return NULL; // Memory allocation failure
    }

    model->type = type; // Set the turbulence model type.

    if (type == TURBULENCE_MODEL_K_EPSILON && params)
    {
        memcpy(&model->params, params, sizeof(KEpsilonParameters)); // Copy parameters for k-epsilon model.
    }
    else
    {
        // Default values for k-epsilon model.
        model->params = (KEpsilonParameters){
            .c_mu = 0.09,
            .sigma_k = 1.0,
            .sigma_eps = 1.3,
            .c1_eps = 1.44,
            .c2_eps = 1.92};
    }
    return model; // Return the created TurbulenceModel object.
}

void turbulence_model_destroy(TurbulenceModel *model)
{
    // Free the resources of the turbulence model.
    if (model)
    {
        free(model); // Free the TurbulenceModel struct.
    }
}

void turb_model_initialize(const TurbulenceModel *model, const Mesh *mesh, FlowState *state)
{
    // Initialize turbulence fields (k, epsilon) in the flow state.
    if (!model || !mesh || !state)
    {
        return; // Invalid input
    }

    size_t N = mesh_get_num_nodes(mesh); // Get number of nodes from mesh.

    for (size_t i = 0; i < N; ++i)
    {
        state->turbulence_kinetic_energy[i] = 1e-3;   // Set initial turbulence kinetic energy.
        state->turbulence_dissipation_rate[i] = 1e-4; // Set initial turbulence dissipation rate.
    }
}

double *turb_model_compute_viscosity(const TurbulenceModel *model, const Mesh *mesh, const FlowState *state)
{
    // Compute turbulent viscosity ν_t at each node.
    if (!model || !mesh || !state || (uintptr_t)model == 1)
    {
        return NULL; // Invalid input
    }

    size_t N = mesh_get_num_nodes(mesh);                 // Get number of nodes from mesh.
    double *nu_t = (double *)malloc(sizeof(double) * N); // Allocate memory for turbulent viscosity.

    // Check for allocation failure.
    if (!nu_t)
    {
        return NULL; // Memory allocation failure
    }

    double c_mu = model->params.c_mu; // Get coefficient for k-epsilon model.

    // Compute turbulent viscosity for each node.
    for (size_t i = 0; i < N; ++i)
    {
        double k = fmax(state->turbulence_kinetic_energy[i], 1e-10);         // Get turbulence kinetic energy.
        double epsilon = fmax(state->turbulence_dissipation_rate[i], 1e-10); // Get turbulence dissipation rate.
        nu_t[i] = c_mu * (k * k) / epsilon;                                  // Compute turbulent viscosity.
    }
    return nu_t; // Return the computed turbulent viscosity array.
}

/// Update the turbulence model in the flow state.
/// This function computes the turbulent viscosity and updates the velocity field.
/// It uses a least-squares method to compute the gradient of the velocity field.
void turb_model_update(
    const TurbulenceModel *model, // Turbulence model to use 
    const Mesh *mesh, // Mesh struct (must contain connectivity, coords, num_cells, nodes_per_cell)
    FlowState *state, // FlowState with nodal velocity (flattened [i*3 + {0,1,2}])
    double dt // Timestep size
) 
{
    // Update the turbulence model (k, epsilon) fields by one time step.
    if (!model || !mesh || !state)
        return; // Invalid input

    size_t N = mesh_get_num_nodes(mesh); // Get number of nodes from mesh.
    size_t C = mesh->num_cells; // Get number of cells from mesh.
    size_t P = mesh->nodes_per_cell; // Get number of nodes per cell from mesh.

    // const double c_mu = model->params.c_mu; // Coefficient for k-epsilon model.
    const double c1_eps = model->params.c1_eps; // Coefficient for the epsilon equation.
    const double c2_eps = model->params.c2_eps; // Coefficient for the epsilon equation.

    // Allocate memory for the new velocity field.
    double *nu_t = turb_model_compute_viscosity(model, mesh, state);

    // Check for allocation failure.
    if (!nu_t) 
        return; // Memory allocation failure

    // Copy old velocity into vel_old.
    size_t *neighbors = malloc(sizeof(size_t) * N); // Allocate memory for neighbors.
    uint8_t *visited = calloc(N, 1); // Allocate memory for visited nodes.

    // Check for allocation failure.
    if (!neighbors || !visited)
    {
        free(nu_t); // Free previously allocated memory.
        free(neighbors); // Free previously allocated memory.
        free(visited); // Free previously allocated memory.
        return; // Memory allocation failure
    }

    // For each node, build least-squares system:
    // 1) Find all neighboring nodes of node i.
    // 2) Build least-squares system for the velocity gradient.
    // 3) Compute the determinant of the system matrix.
    // 4) Compute the inverse of the system matrix.
    // 5) Compute the convective acceleration a = -(u·∇)u.
    // 6) Update velocities.
    // 7) Free allocated memory.
    // 8) Return the updated velocity field. 
    for (size_t i = 0; i < N; ++i)
    {
        memset(visited, 0, N); // Reset visited array.
        size_t nb_count = 0; // Initialize neighbor count.

        // Find all neighboring nodes of node i
        for (size_t c = 0; c < C; ++c)
        {
            // Get the connectivity for cell c.
            const size_t *conn = mesh->connectivity + c * P;

            // Check membership.
            int in_cell = 0;

            // Check if node i is in the cell.
            for (size_t k = 0; k < P; ++k)
            {
                // Check if node i is in the cell.
                if (conn[k] == i)
                {
                    in_cell = 1; // Set in_cell flag.
                    break; // Break if found.
                }
            }

            if (!in_cell) // Check if node i is not in the cell.
                continue; // Skip if node i is not in the cell.

            // Add all other nodes from this cell.
            // Iterate through nodes in the cell.
            for (size_t k = 0; k < P; ++k)
            {
                size_t j = conn[k]; // Get node index.

                // Check if node j is the same as node i or if it has already been visited.
                if (j == i || visited[j])
                    continue; // Skip if node is itself or already visited.

                // Mark node j as visited and add it to the neighbors list.
                visited[j] = 1; // Mark node as visited.
                neighbors[nb_count++] = j; // Add node to neighbors list.
            }
        }

        // Skip if too few neighbors to form a 3 × 3 system.
        if (nb_count < 3)
            continue; // Skip to next node.

        double AtA[3][3] = {{0}}, Atb[3][3] = {{0}}; // Initialize matrices.

        // 2) Assemble AtA and Atb for least-squares: AtA ∈ R^{3×3}, Atb ∈ R^{3×3}
        double xi[3] = {
            mesh->coordinates[3 * i + 0], // Node i's coordinates
            mesh->coordinates[3 * i + 1], // Node i's coordinates
            mesh->coordinates[3 * i + 2]}; // Node i's coordinates

        // Node i’s velocity.
        double ui[3] = {
            state->velocity[3 * i + 0], // Node i's velocity
            state->velocity[3 * i + 1], //
            state->velocity[3 * i + 2]};

        // Iterate through neighbors.
        for (size_t n = 0; n < nb_count; ++n)
        {
            size_t j = neighbors[n]; // Get neighbor index.

            // Node j's coordinates.
            double r[3] = {
                mesh->coordinates[3 * j + 0] - xi[0],
                mesh->coordinates[3 * j + 1] - xi[1],
                mesh->coordinates[3 * j + 2] - xi[2]};

            // Node j's velocity.
            double du[3] = {
                state->velocity[3 * j + 0] - ui[0],
                state->velocity[3 * j + 1] - ui[1],
                state->velocity[3 * j + 2] - ui[2]};

            // Accumulate AtA[k][l] += r[k]*r[l]
            for (int l = 0; l < 3; ++l)
            {
                // Accumulate AtA[k][l] += r[k]*r[l]
                for (int m = 0; m < 3; ++m)
                {
                    AtA[l][m] += r[l] * r[m]; // X component of AtA.
                    Atb[l][m] += r[l] * du[m]; // Y component of AtA.
                }
            }
        }

        // 3) Compute determinant of AtA.
        double det =
            AtA[0][0] * (AtA[1][1] * AtA[2][2] - AtA[1][2] * AtA[2][1]) -
            AtA[0][1] * (AtA[1][0] * AtA[2][2] - AtA[1][2] * AtA[2][0]) +
            AtA[0][2] * (AtA[1][0] * AtA[2][1] - AtA[1][1] * AtA[2][0]);

        // Check for singular matrix (det = 0).
        if (fabs(det) < 1e-12)
            continue; // Skip to next node.

        // 4) Compute inverse of AtA using Cramer's rule.
        double invA[3][3];
        invA[0][0] = (AtA[1][1] * AtA[2][2] - AtA[1][2] * AtA[2][1]) / det;
        invA[0][1] = -(AtA[0][1] * AtA[2][2] - AtA[0][2] * AtA[2][1]) / det;
        invA[0][2] = (AtA[0][1] * AtA[1][2] - AtA[0][2] * AtA[1][1]) / det;
        invA[1][0] = -(AtA[1][0] * AtA[2][2] - AtA[1][2] * AtA[2][0]) / det;
        invA[1][1] = (AtA[0][0] * AtA[2][2] - AtA[0][2] * AtA[2][0]) / det;
        invA[1][2] = -(AtA[0][0] * AtA[1][2] - AtA[0][2] * AtA[1][0]) / det;
        invA[2][0] = (AtA[1][0] * AtA[2][1] - AtA[1][1] * AtA[2][0]) / det;
        invA[2][1] = -(AtA[0][0] * AtA[2][1] - AtA[0][1] * AtA[2][0]) / det;
        invA[2][2] = (AtA[0][0] * AtA[1][1] - AtA[0][1] * AtA[1][0]) / det;

        // Compute gradient: grad[l][m] = ∑_k invA[l][k] * Atb[k][m]
        double grad[3][3] = {{0}};
        
        // Compute gradient using the inverse of AtA and Atb.
        for (int l = 0; l < 3; ++l)
            // Iterate through gradient dimensions.
            for (int m = 0; m < 3; ++m)
                // Iterate through gradient dimensions.
                for (int k = 0; k < 3; ++k)
                    grad[l][m] += invA[l][k] * Atb[k][m]; // Accumulate gradient component.

        // 5) Compute convective acceleration a = -(u·∇)u
        double velocity_grad_mag2 = 0.0;
        
        // Compute convective acceleration using the gradient.
        for (int l = 0; l < 3; ++l)
            // Iterate through gradient dimensions.
            for (int m = 0; m < 3; ++m)
                velocity_grad_mag2 += grad[l][m] * grad[l][m]; // Accumulate gradient component.

        // 6) Compute the turbulent viscosity.
        double Pk = nu_t[i] * velocity_grad_mag2;

        // 7) Compute the new velocity.
        double k = fmax(state->turbulence_kinetic_energy[i], 1e-10); // Get turbulence kinetic energy.
        double eps = fmax(state->turbulence_dissipation_rate[i], 1e-10); // Get turbulence dissipation rate.


        state->turbulence_kinetic_energy[i] += dt * (Pk - eps); // Update turbulence kinetic energy.

        double eps_prod = (eps / k) * (c1_eps * Pk - c2_eps * eps); // Update turbulence dissipation rate.
        
        // Update turbulence dissipation rate.
        state->turbulence_dissipation_rate[i] += dt * eps_prod;

        state->turbulence_kinetic_energy[i] = fmax(state->turbulence_kinetic_energy[i], 1e-10); // Ensure non-negative.
        state->turbulence_dissipation_rate[i] = fmax(state->turbulence_dissipation_rate[i], 1e-10); // Ensure non-negative.
    }

    free(nu_t); // Free turbulent viscosity array. 
    free(neighbors); // Free neighbors array.
    free(visited); // Free visited array. 
}
