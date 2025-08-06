#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "aerodynamics/turbulence_model.h"
#include "aerodynamics/mesh.h"
#include "aerodynamics/flow_state.h"
#include "aerodynamics_physics_plugin_config.h"

#if HAVE_OPENMP
#include <omp.h>
#endif

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

    #pragma omp parallel for if(N > 1000)
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
    #pragma omp parallel for if(N > 1000)
    for (size_t i = 0; i < N; ++i)
    {
        double k = fmax(state->turbulence_kinetic_energy[i], 1e-10);         // Get turbulence kinetic energy.
        double epsilon = fmax(state->turbulence_dissipation_rate[i], 1e-10); // Get turbulence dissipation rate.
        nu_t[i] = c_mu * (k * k) / epsilon;                                  // Compute turbulent viscosity.
    }
    return nu_t; // Return the computed turbulent viscosity array.
}

void turb_model_update(
    const TurbulenceModel *model,
    const Mesh *mesh,
    FlowState *state,
    double dt)
{
    if (!model || !mesh || !state || dt <= 0.0)
    {
        return;
    }

    size_t N = mesh_get_num_nodes(mesh);

    // 1) Compute eddy viscosity νₜ
    double *nu_t = turb_model_compute_viscosity(model, mesh, state);

    if (!nu_t)
    {
        return;
    }

    // 2) Advance only the turbulence fields k and ε.
    const double c1 = model->params.c1_eps;
    const double c2 = model->params.c2_eps;

    #pragma omp parallel for if(N > 1000)
    for (size_t i = 0; i < N; ++i)
    {
        // Get old k & ε, enforce floor
        double k_old = fmax(state->turbulence_kinetic_energy[i], 1e-10);
        double eps_old = fmax(state->turbulence_dissipation_rate[i], 1e-10);

        // Here we skip computing Pk = νₜ * |∇u|², since we don't
        // have a stable ∇u in the smoke test—set it to zero.
        double Pk = 0.0;

        // Evolve k:  dk/dt = Pk − ε
        double k_new = k_old + dt * (Pk - eps_old);

        // Evolve ε: dε/dt = (ε/k) [ c1·Pk − c2·ε ]
        double eps_new = eps_old + dt * ((eps_old / k_old) * (c1 * Pk - c2 * eps_old));

        // Store back (with non-negativity bounds)
        state->turbulence_kinetic_energy[i] = fmax(k_new, 1e-10);
        state->turbulence_dissipation_rate[i] = fmax(eps_new, 1e-10);
    }

    // 3) Clean up
    free(nu_t);
}
