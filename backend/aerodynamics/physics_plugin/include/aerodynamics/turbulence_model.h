#ifndef AERODYNAMICS_TURBULENCE_MODEL_H
#define AERODYNAMICS_TURBULENCE_MODEL_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <stddef.h>                  // for size_t
#include "aerodynamics/mesh.h"       // For Mesh
#include "aerodynamics/flow_state.h" // For FlowState

    /**
     * Turbulence models supported by the solver.
     */
    typedef enum
    {
        TURBULENCE_MDOEL_NONE = 0,   // No turbulence model
        TURBULENCE_MODEL_K_EPSILON,  // k-epsilon turbulence model
        TURBULENCE_MODEL_SMAGORINSKY // Smagorinsky turbulence model
    } TurbulenceModelType;

    /**
     * Parameters for the k-epsilon turbulence model.
     */
    typedef struct
    {
        double c_mu;      // Coefficient for the k-epsilon model.
        double sigma_k;   // Turbulence kinetic energy coefficient.
        double sigma_eps; // Turbulence dissipation rate coefficient.
        double c1_eps;    // Coefficient for the epsilon equation.
        double c2_eps;    // Coefficient for the epsilon equation.
    } KEpsilonParameters; // KEpsilonParameters structure

    /**
     * General turbulence model structure.
     */
    typedef struct TurbulenceModel
    {
        TurbulenceModelType type;  // Type of turbulence model.
        KEpsilonParameters params; // Parameters for the k-epsilon model.
    } TurbulenceModel;             // TurbulenceModel structure

    /**
     * Allocate and initialize a turbulence model.
     */
    TurbulenceModel *turbulence_model_create(TurbulenceModelType type, const KEpsilonParameters *params);

    /**
     * Free resources of a turbulence model.
     */
    void turbulence_model_destroy(TurbulenceModel *model);

    /**
     * Initialize turbulence fields (k, epsilon) in the flow state.
     */
    void turb_model_initialize(const TurbulenceModel *model, const Mesh *mesh, FlowState *state);

    /**
     * Compute turbulennt viscosity ν_t at each node.
     *
     * Caller must free returned array.
     *
     * @return double* array [num_nodes] of turbulent viscosity values.
     */
    double *turb_model_compute_visocsity(const TurbulenceModel *model, const Mesh *mesh, const FlowState *state);

    /**
     *  Advance turbulence model (k, epsilon) fields by one time step.
     */
    void turb_model_update(const TurbulenceModel *model, const Mesh *mesh, FlowState *state, double dt);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // AERODYNAMICS_TURBULENCE_MODEL_H