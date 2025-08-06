#ifndef AERODYNAMICS_BINDINGS_H
#define AERODYNAMICS_BINDINGS_H

#ifdef __cplusplus
extern "C"
{
#endif

#include "aerodynamics/mesh.h"
#include "aerodynamics/turbulence_model.h"
#include "aerodynamics/flow_state.h"
#include "aerodynamics/actuator.h"
#include "aerodynamics/solver.h"

    /**
     * Opaque handles for external bindings.
     */
    typedef void *MeshHandle;
    typedef void *TurbulenceModelHandle;
    typedef void *ActuatorHandle;
    typedef void *SolverHandle;
    typedef void *FlowStateHandle;

    /**
     * Mesh bindings.
     */
    MeshHandle mesh_create_bind(size_t num_nodes, const double *coords, size_t num_cells, size_t nodes_per_cell, const size_t *connectivity);
    void mesh_destroy_bind(MeshHandle mesh);

    /**
     * Turbulence model bindings.
     */
    TurbulenceModelHandle turb_model_create_bind(double c_mu, double sigma_k, double sigma_eps, double c1_eps, double c2_eps);
    void turb_model_destroy_bind(TurbulenceModelHandle turb_model);

    /**
     * Actuator bindings.
     */
    ActuatorHandle actuator_create_bind(const char *name, int type, size_t node_count, const size_t *node_ids, const double direction[3], double gain);
    void actuator_set_command_bind(ActuatorHandle act, double command);
    void actuator_destroy_bind(ActuatorHandle act);

    /**
     * Solver bindings.
     */
    SolverHandle solver_create_bind(MeshHandle mesh, TurbulenceModelHandle turb_model);
    void solver_initialize_bind(SolverHandle solver);
    void solver_step_bind(SolverHandle solver, double dt);
    void solver_apply_actuator_bind(SolverHandle solver, ActuatorHandle actuator, double dt);
    void solver_destroy_bind(SolverHandle solver);

    /**
     * FlowState bindings.
     */
    FlowStateHandle flow_state_create_bind(MeshHandle mesh);
    void solver_read_state_bind(SolverHandle solver, FlowStateHandle state);
    void flow_state_destroy_bind(FlowStateHandle state);

    // Data extraction functions
    void flow_state_get_velocity_bind(FlowStateHandle state, double *out_velocity);
    void flow_state_get_pressure_bind(FlowStateHandle state, double *out_pressure);
    void flow_state_get_tke_bind(FlowStateHandle state, double *out_tke);
    void flow_state_get_dissipation_bind(FlowStateHandle state, double *out_eps);

    /**
     * VTK writer bindings.
     */
    int vtk_write_solution_bind(const char *filename, MeshHandle mesh, FlowStateHandle state, int format);
    int vtk_write_mesh_bind(const char *filename, MeshHandle mesh, int format);
    void *vtk_create_time_series_writer_bind(const char *base_filename, int format);
    int vtk_write_timestep_bind(void *writer, int timestep, double time, MeshHandle mesh, FlowStateHandle state);
    void vtk_close_time_series_writer_bind(void *writer);

#ifdef __cplusplus
}
#endif

#endif // AERODYNAMICS_BINDINGS_H
