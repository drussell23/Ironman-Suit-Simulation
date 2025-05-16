#ifndef AERODYNAMICS_SOLVER_H
#define AERODYNAMICS_SOLVER_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <stddef.h>   // for size_t

#include "aerodynamics/mesh.h" // your Mesh definition
#include "aerodynamics/turbulence_model.h"  // your TurbulenceModel definition
#include "aerodynamics/flow_state.h"        // your FlowState definition
#include "aerodynamics/actuator.h"          // your Actuator definition

    /**
     * Opaque handle for the CFD solver.
     */
    typedef struct Solver Solver;

    /**
     * @brief Construct a new Solver.
     *
     * Takes ownership of neither mesh nor turb_model; they must outlive the Solver.
     *
     * @param mesh          Pointer to a preconfigured Mesh.
     * @param turb_model    Pointer to an initialized TurbulenceModel.
     * @return Solver*      New solver instance (NULL on failure).
     */
    Solver *solver_create(Mesh *mesh, TurbulenceModel *turb_model);

    /**
     * @brief Destroy a Solver, freeing its internal resources.
     *
     * @param solver  Solver instance to destroy.
     */
    void solver_destroy(Solver *solver);

    /**
     * @brief Initialize solver internals (allocate fields, set BCs).
     *
     * Must be called once before any calls to solver_step().
     *
     * @param solver  Solver instance.
     */
    void solver_initialize(Solver *solver);

    /**
     * @brief Read current flow state into user-provided structure.
     *
     * @param solver  Solver instance.
     * @param out     Pointer to FlowState struct to populate.
     */
    void solver_read_state(const Solver *solver, FlowState *out);

    /**
     * @brief Apply actuator commands to the solver.
     *
     * @param solver     Solver instance.
     * @param acts       Array of Actuator objects.
     * @param act_count  Number of actuators in the array.
     */
    void solver_apply_actuators(Solver *solver,
                                const Actuator *acts,
                                size_t act_count, double dt);

    /**
     * @brief Advance the solution by one time step.
     *
     * @param solver  Solver instance.
     * @param dt      Time step size (seconds).
     */
    void solver_step(Solver *solver, double dt);

#ifdef __cplusplus
}
#endif

#endif // AERODYNAMICS_SOLVER_H
