# File: examples/CMakeLists.txt

cmake_minimum_required(VERSION 3.10)
project(FullPipelineExample C)
set(CMAKE_C_STANDARD 99)

#
# 1. Tell CMake where to look for headers and libraries relative to this folder
#

# Assume this CMakeLists.txt lives in:
#   physics_plugin/examples/CMakeLists.txt
#
# Then:
#   - Headers are in    physics_plugin/include/
#   - Shared lib is in  physics_plugin/build/libaerodynamics_physics_plugin.dylib
#
# We compute those two paths here:

# Get path to “physics_plugin/examples/”:

# Paths relative to this examples/ directory
set(PLUGIN_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(AERO_INCLUDE_DIR ${PLUGIN_ROOT}/include)
set(AERO_LIBRARY_DIR ${PLUGIN_ROOT}/build)

# Include aerodynamics headers
include_directories(${AERO_INCLUDE_DIR})

# Locate the aerodynamics physics plugin library
find_library(AERO_LIB
    NAMES aerodynamics_physics_plugin
    PATHS ${AERO_LIBRARY_DIR}
    NO_DEFAULT_PATH)
if(NOT AERO_LIB)
    message(FATAL_ERROR "Could not find aerodynamics_physics_plugin in ${AERO_LIBRARY_DIR}")
endif()

# Build the example executable
add_executable(full_pipeline_example full_pipeline_example.c)

target_link_libraries(full_pipeline_example PRIVATE ${AERO_LIB})

# On macOS, set RPATH so the shared lib can be found at runtime
if(APPLE)
    set_target_properties(full_pipeline_example PROPERTIES
        INSTALL_RPATH "@loader_path/../../build"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()
