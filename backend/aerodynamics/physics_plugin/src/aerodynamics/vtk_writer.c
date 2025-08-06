#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "aerodynamics/vtk_writer.h"
#include "aerodynamics/mesh.h"
#include "aerodynamics/flow_state.h"

/**
 * @brief Time series writer state
 */
typedef struct {
    char *base_filename;
    VTKFormat format;
    FILE *pvd_file;  // ParaView Data file for time series
    int timestep_count;
} VTKTimeSeriesWriter;

/**
 * @brief Get file extension for VTK format
 */
static const char* get_vtk_extension(VTKFormat format) {
    switch (format) {
        case VTK_FORMAT_ASCII:
        case VTK_FORMAT_BINARY:
            return ".vtk";
        case VTK_FORMAT_XML:
            return ".vtu";  // Unstructured grid
        default:
            return ".vtk";
    }
}

/**
 * @brief Write legacy VTK header
 */
static void write_vtk_header(FILE *file, const char *title, VTKFormat format) {
    fprintf(file, "# vtk DataFile Version 3.0\n");
    fprintf(file, "%s\n", title);
    fprintf(file, "%s\n", format == VTK_FORMAT_ASCII ? "ASCII" : "BINARY");
    fprintf(file, "DATASET UNSTRUCTURED_GRID\n");
}

/**
 * @brief Write mesh points in legacy VTK format
 */
static void write_vtk_points(FILE *file, const Mesh *mesh) {
    size_t n = mesh->num_nodes;
    fprintf(file, "POINTS %zu double\n", n);
    
    for (size_t i = 0; i < n; ++i) {
        fprintf(file, "%.6e %.6e %.6e\n",
                mesh->coordinates[3*i + 0],
                mesh->coordinates[3*i + 1],
                mesh->coordinates[3*i + 2]);
    }
}

/**
 * @brief Get VTK cell type ID
 */
static int get_vtk_cell_type(size_t nodes_per_cell) {
    switch (nodes_per_cell) {
        case 4: return 10;  // VTK_TETRA
        case 8: return 12;  // VTK_HEXAHEDRON
        case 6: return 13;  // VTK_WEDGE
        case 5: return 14;  // VTK_PYRAMID
        default: return -1;
    }
}

/**
 * @brief Write mesh cells in legacy VTK format
 */
static void write_vtk_cells(FILE *file, const Mesh *mesh) {
    size_t c = mesh->num_cells;
    size_t p = mesh->nodes_per_cell;
    size_t total_size = c * (p + 1);  // +1 for count per cell
    
    fprintf(file, "CELLS %zu %zu\n", c, total_size);
    
    for (size_t i = 0; i < c; ++i) {
        fprintf(file, "%zu", p);
        for (size_t j = 0; j < p; ++j) {
            fprintf(file, " %zu", mesh->connectivity[i*p + j]);
        }
        fprintf(file, "\n");
    }
    
    // Cell types
    fprintf(file, "CELL_TYPES %zu\n", c);
    int cell_type = get_vtk_cell_type(p);
    for (size_t i = 0; i < c; ++i) {
        fprintf(file, "%d\n", cell_type);
    }
}

/**
 * @brief Write point data in legacy VTK format
 */
static void write_vtk_point_data(FILE *file, const FlowState *state, size_t num_nodes) {
    fprintf(file, "\nPOINT_DATA %zu\n", num_nodes);
    
    // Velocity vector field
    fprintf(file, "VECTORS velocity double\n");
    for (size_t i = 0; i < num_nodes; ++i) {
        fprintf(file, "%.6e %.6e %.6e\n",
                state->velocity[3*i + 0],
                state->velocity[3*i + 1],
                state->velocity[3*i + 2]);
    }
    
    // Pressure scalar field
    fprintf(file, "\nSCALARS pressure double 1\n");
    fprintf(file, "LOOKUP_TABLE default\n");
    for (size_t i = 0; i < num_nodes; ++i) {
        fprintf(file, "%.6e\n", state->pressure[i]);
    }
    
    // Turbulence kinetic energy
    if (state->turbulence_kinetic_energy) {
        fprintf(file, "\nSCALARS turbulence_kinetic_energy double 1\n");
        fprintf(file, "LOOKUP_TABLE default\n");
        for (size_t i = 0; i < num_nodes; ++i) {
            fprintf(file, "%.6e\n", state->turbulence_kinetic_energy[i]);
        }
    }
    
    // Turbulence dissipation rate
    if (state->turbulence_dissipation_rate) {
        fprintf(file, "\nSCALARS turbulence_dissipation_rate double 1\n");
        fprintf(file, "LOOKUP_TABLE default\n");
        for (size_t i = 0; i < num_nodes; ++i) {
            fprintf(file, "%.6e\n", state->turbulence_dissipation_rate[i]);
        }
    }
    
    // Velocity magnitude (derived quantity)
    fprintf(file, "\nSCALARS velocity_magnitude double 1\n");
    fprintf(file, "LOOKUP_TABLE default\n");
    for (size_t i = 0; i < num_nodes; ++i) {
        double vx = state->velocity[3*i + 0];
        double vy = state->velocity[3*i + 1];
        double vz = state->velocity[3*i + 2];
        double vmag = sqrt(vx*vx + vy*vy + vz*vz);
        fprintf(file, "%.6e\n", vmag);
    }
}

/**
 * @brief Write XML VTU format header
 */
static void write_vtu_header(FILE *file) {
    fprintf(file, "<?xml version=\"1.0\"?>\n");
    fprintf(file, "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
    fprintf(file, "  <UnstructuredGrid>\n");
}

/**
 * @brief Write XML VTU format footer
 */
static void write_vtu_footer(FILE *file) {
    fprintf(file, "  </UnstructuredGrid>\n");
    fprintf(file, "</VTKFile>\n");
}

/**
 * @brief Write mesh and solution in XML VTU format
 */
static int write_vtu_format(FILE *file, const Mesh *mesh, const FlowState *state) {
    size_t n = mesh->num_nodes;
    size_t c = mesh->num_cells;
    size_t p = mesh->nodes_per_cell;
    
    write_vtu_header(file);
    
    fprintf(file, "    <Piece NumberOfPoints=\"%zu\" NumberOfCells=\"%zu\">\n", n, c);
    
    // Points
    fprintf(file, "      <Points>\n");
    fprintf(file, "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n");
    for (size_t i = 0; i < n; ++i) {
        fprintf(file, "          %.6e %.6e %.6e\n",
                mesh->coordinates[3*i + 0],
                mesh->coordinates[3*i + 1],
                mesh->coordinates[3*i + 2]);
    }
    fprintf(file, "        </DataArray>\n");
    fprintf(file, "      </Points>\n");
    
    // Cells
    fprintf(file, "      <Cells>\n");
    
    // Connectivity
    fprintf(file, "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n");
    for (size_t i = 0; i < c; ++i) {
        fprintf(file, "          ");
        for (size_t j = 0; j < p; ++j) {
            fprintf(file, "%zu ", mesh->connectivity[i*p + j]);
        }
        fprintf(file, "\n");
    }
    fprintf(file, "        </DataArray>\n");
    
    // Offsets
    fprintf(file, "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n");
    for (size_t i = 0; i < c; ++i) {
        fprintf(file, "          %zu\n", (i+1)*p);
    }
    fprintf(file, "        </DataArray>\n");
    
    // Types
    fprintf(file, "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n");
    int cell_type = get_vtk_cell_type(p);
    for (size_t i = 0; i < c; ++i) {
        fprintf(file, "          %d\n", cell_type);
    }
    fprintf(file, "        </DataArray>\n");
    
    fprintf(file, "      </Cells>\n");
    
    // Point data
    if (state) {
        fprintf(file, "      <PointData Vectors=\"velocity\" Scalars=\"pressure\">\n");
        
        // Velocity
        fprintf(file, "        <DataArray type=\"Float64\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"ascii\">\n");
        for (size_t i = 0; i < n; ++i) {
            fprintf(file, "          %.6e %.6e %.6e\n",
                    state->velocity[3*i + 0],
                    state->velocity[3*i + 1],
                    state->velocity[3*i + 2]);
        }
        fprintf(file, "        </DataArray>\n");
        
        // Pressure
        fprintf(file, "        <DataArray type=\"Float64\" Name=\"pressure\" format=\"ascii\">\n");
        for (size_t i = 0; i < n; ++i) {
            fprintf(file, "          %.6e\n", state->pressure[i]);
        }
        fprintf(file, "        </DataArray>\n");
        
        // Turbulence fields
        if (state->turbulence_kinetic_energy) {
            fprintf(file, "        <DataArray type=\"Float64\" Name=\"turbulence_kinetic_energy\" format=\"ascii\">\n");
            for (size_t i = 0; i < n; ++i) {
                fprintf(file, "          %.6e\n", state->turbulence_kinetic_energy[i]);
            }
            fprintf(file, "        </DataArray>\n");
        }
        
        if (state->turbulence_dissipation_rate) {
            fprintf(file, "        <DataArray type=\"Float64\" Name=\"turbulence_dissipation_rate\" format=\"ascii\">\n");
            for (size_t i = 0; i < n; ++i) {
                fprintf(file, "          %.6e\n", state->turbulence_dissipation_rate[i]);
            }
            fprintf(file, "        </DataArray>\n");
        }
        
        // Velocity magnitude
        fprintf(file, "        <DataArray type=\"Float64\" Name=\"velocity_magnitude\" format=\"ascii\">\n");
        for (size_t i = 0; i < n; ++i) {
            double vx = state->velocity[3*i + 0];
            double vy = state->velocity[3*i + 1];
            double vz = state->velocity[3*i + 2];
            double vmag = sqrt(vx*vx + vy*vy + vz*vz);
            fprintf(file, "          %.6e\n", vmag);
        }
        fprintf(file, "        </DataArray>\n");
        
        fprintf(file, "      </PointData>\n");
    }
    
    fprintf(file, "    </Piece>\n");
    write_vtu_footer(file);
    
    return 0;
}

int vtk_write_solution(
    const char *filename,
    const Mesh *mesh,
    const FlowState *state,
    VTKFormat format
) {
    if (!filename || !mesh || !state) {
        return -1;
    }
    
    // Construct full filename with extension
    char full_filename[512];
    snprintf(full_filename, sizeof(full_filename), "%s%s", 
             filename, get_vtk_extension(format));
    
    FILE *file = fopen(full_filename, "w");
    if (!file) {
        fprintf(stderr, "vtk_write_solution: Failed to open file '%s'\n", full_filename);
        return -1;
    }
    
    int result = 0;
    
    if (format == VTK_FORMAT_XML) {
        result = write_vtu_format(file, mesh, state);
    } else {
        // Legacy format
        write_vtk_header(file, "Aerodynamics simulation output", format);
        write_vtk_points(file, mesh);
        write_vtk_cells(file, mesh);
        write_vtk_point_data(file, state, mesh->num_nodes);
    }
    
    fclose(file);
    
    if (result == 0) {
        printf("VTK file written: %s\n", full_filename);
    }
    
    return result;
}

int vtk_write_mesh(
    const char *filename,
    const Mesh *mesh,
    VTKFormat format
) {
    if (!filename || !mesh) {
        return -1;
    }
    
    // Construct full filename with extension
    char full_filename[512];
    snprintf(full_filename, sizeof(full_filename), "%s%s", 
             filename, get_vtk_extension(format));
    
    FILE *file = fopen(full_filename, "w");
    if (!file) {
        fprintf(stderr, "vtk_write_mesh: Failed to open file '%s'\n", full_filename);
        return -1;
    }
    
    int result = 0;
    
    if (format == VTK_FORMAT_XML) {
        result = write_vtu_format(file, mesh, NULL);
    } else {
        // Legacy format
        write_vtk_header(file, "Mesh only", format);
        write_vtk_points(file, mesh);
        write_vtk_cells(file, mesh);
    }
    
    fclose(file);
    
    if (result == 0) {
        printf("VTK mesh file written: %s\n", full_filename);
    }
    
    return result;
}

void* vtk_create_time_series_writer(
    const char *base_filename,
    VTKFormat format
) {
    if (!base_filename) {
        return NULL;
    }
    
    VTKTimeSeriesWriter *writer = malloc(sizeof(VTKTimeSeriesWriter));
    if (!writer) {
        return NULL;
    }
    
    writer->base_filename = strdup(base_filename);
    writer->format = format;
    writer->timestep_count = 0;
    
    // Create PVD file for ParaView
    char pvd_filename[512];
    snprintf(pvd_filename, sizeof(pvd_filename), "%s.pvd", base_filename);
    
    writer->pvd_file = fopen(pvd_filename, "w");
    if (!writer->pvd_file) {
        free(writer->base_filename);
        free(writer);
        return NULL;
    }
    
    // Write PVD header
    fprintf(writer->pvd_file, "<?xml version=\"1.0\"?>\n");
    fprintf(writer->pvd_file, "<VTKFile type=\"Collection\" version=\"0.1\">\n");
    fprintf(writer->pvd_file, "  <Collection>\n");
    
    return writer;
}

int vtk_write_timestep(
    void *writer_ptr,
    int timestep,
    double time,
    const Mesh *mesh,
    const FlowState *state
) {
    if (!writer_ptr || !mesh || !state) {
        return -1;
    }
    
    VTKTimeSeriesWriter *writer = (VTKTimeSeriesWriter *)writer_ptr;
    
    // Create timestep filename
    char timestep_filename[512];
    snprintf(timestep_filename, sizeof(timestep_filename), 
             "%s_%06d", writer->base_filename, timestep);
    
    // Write the data file
    int result = vtk_write_solution(timestep_filename, mesh, state, writer->format);
    if (result != 0) {
        return result;
    }
    
    // Add entry to PVD file
    fprintf(writer->pvd_file, "    <DataSet timestep=\"%.6e\" file=\"%s_%06d%s\"/>\n",
            time, writer->base_filename, timestep, get_vtk_extension(writer->format));
    
    writer->timestep_count++;
    
    return 0;
}

void vtk_close_time_series_writer(void *writer_ptr) {
    if (!writer_ptr) {
        return;
    }
    
    VTKTimeSeriesWriter *writer = (VTKTimeSeriesWriter *)writer_ptr;
    
    if (writer->pvd_file) {
        // Write PVD footer
        fprintf(writer->pvd_file, "  </Collection>\n");
        fprintf(writer->pvd_file, "</VTKFile>\n");
        fclose(writer->pvd_file);
        
        printf("Time series written: %s.pvd (%d timesteps)\n", 
               writer->base_filename, writer->timestep_count);
    }
    
    free(writer->base_filename);
    free(writer);
}
