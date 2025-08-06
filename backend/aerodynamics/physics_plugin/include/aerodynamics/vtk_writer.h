#ifndef AERODYNAMICS_VTK_WRITER_H
#define AERODYNAMICS_VTK_WRITER_H

#include <stddef.h>

// Forward declarations
typedef struct Mesh Mesh;
typedef struct FlowState FlowState;

/**
 * @brief VTK file format options
 */
typedef enum {
    VTK_FORMAT_ASCII,      /**< Human-readable ASCII format */
    VTK_FORMAT_BINARY,     /**< Binary format (more compact) */
    VTK_FORMAT_XML         /**< Modern XML-based VTK format */
} VTKFormat;

/**
 * @brief Write mesh and flow state to VTK file for visualization
 * 
 * @param filename Output filename (without extension)
 * @param mesh Computational mesh
 * @param state Flow state with velocity, pressure, etc.
 * @param format VTK file format
 * @return 0 on success, -1 on error
 */
int vtk_write_solution(
    const char *filename,
    const Mesh *mesh,
    const FlowState *state,
    VTKFormat format
);

/**
 * @brief Write mesh only to VTK file
 * 
 * @param filename Output filename (without extension)
 * @param mesh Computational mesh
 * @param format VTK file format
 * @return 0 on success, -1 on error
 */
int vtk_write_mesh(
    const char *filename,
    const Mesh *mesh,
    VTKFormat format
);

/**
 * @brief Create a time series writer for transient simulations
 * 
 * @param base_filename Base filename for series
 * @param format VTK file format
 * @return Handle to time series writer, or NULL on error
 */
void* vtk_create_time_series_writer(
    const char *base_filename,
    VTKFormat format
);

/**
 * @brief Add a timestep to the time series
 * 
 * @param writer Time series writer handle
 * @param timestep Current timestep number
 * @param time Physical time
 * @param mesh Computational mesh
 * @param state Flow state
 * @return 0 on success, -1 on error
 */
int vtk_write_timestep(
    void *writer,
    int timestep,
    double time,
    const Mesh *mesh,
    const FlowState *state
);

/**
 * @brief Close and finalize time series writer
 * 
 * @param writer Time series writer handle
 */
void vtk_close_time_series_writer(void *writer);

#endif /* AERODYNAMICS_VTK_WRITER_H */
