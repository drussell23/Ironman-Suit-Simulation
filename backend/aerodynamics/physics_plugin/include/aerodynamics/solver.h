#ifndef AERODYNAMICS_SOLVER_H
#define AERODYNAMICS_SOLVER_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <stddef.h>                        // for size_t
#include "aerodynamics/mesh.h"             // Mesh definition
#include "aerodynamics/turbulence_model.h" // TurbulenceModel definition
#include "aerodynamics/flow_state.h"       // FlowState definition
#include "aerodynamics/actuator.h"         // Actuator definition

    /**
     * Opaque solver type.
     */
    typedef struct Solver Solver;

    /**
     * Create a new Solver instance.
     * @param mesh        Preconfigured Mesh (must outlive Solver).
     * @param turb_model  Initialized TurbulenceModel (must outlive Solver).
     * @return New Solver*, or NULL on error.
     */
    Solver *solver_create(Mesh *mesh,
                          TurbulenceModel *turb_model);

    /**
     * Free a Solver and its internal resources.
     * Does NOT free mesh or turb_model.
     */
    void solver_destroy(Solver *solver);

    /**
     * Initialize solver internals:
     *  - Allocate and init internal FlowState
     *  - Initialize mesh and turbulence model
     * Must be called once before solver_step() or apply_actuators().
     */
    void solver_initialize(Solver *solver);

    /**
     * Copy current internal FlowState into user-provided FlowState.
     * @param solver  Solver instance
     * @param out      Pre-allocated FlowState (by flow_state_create or similar)
     */
    void solver_read_state(const Solver *solver, FlowState *out);

    /**
     * Apply an array of actuators to solver's internal state.
     * @param solver      Solver instance
     * @param acts        Array of Actuator objects
     * @param act_count   Number of actuators in array
     * @param dt          Timestep size (seconds)
     */
    void solver_apply_actuators(Solver *solver,
                                const Actuator *acts,
                                size_t act_count,
                                double dt);

    /**
     * Advance solution one time step.
     * @param solver  Solver instance
     * @param dt      Timestep size (seconds)
     */
    void solver_step(Solver *solver, double dt);

#ifdef __cplusplus
}
#endif

#endif // AERODYNAMICS_SOLVER_H
