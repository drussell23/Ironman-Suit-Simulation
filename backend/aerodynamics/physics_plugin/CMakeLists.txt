# Next, rebuild and rerun tests:
# ------------------------------
# rm -rf build && mkdir build && cd build
# cmake -DENABLE_TESTING=ON ..
# cmake --build . -- -j4
# ctest --output-on-failure

cmake_minimum_required(VERSION 3.18)

# turn on compile_commands.json generation
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Project Setup (C-only)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
project(aerodynamics_physics_plugin
  VERSION 1.0.0
  DESCRIPTION "Aerodynamics physics plugin (CFD solver + turbulence models)"
  LANGUAGES C
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# User Options
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
option(BUILD_SHARED_LIBS "Build as a shared library (.so/.dll)" ON)
option(ENABLE_TESTING "Enable building and running unit tests" ON)
option(ENABLE_EXAMPLES "Enable building example executables" OFF)
option(ENABLE_COVERAGE "Enable code-coverage instrumentation" OFF)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Compiler Standards & Flags
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED TRUE)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

cmake_policy(SET CMP0091 NEW) # Keep POSITION_INDEPENDENT_CODE behavior modern

# → We’ll scope -Werror to just our library target, below.
if(ENABLE_COVERAGE)
  if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(--coverage -O0)
    add_link_options(--coverage)
  else()
    message(WARNING "Code coverage not supported for this compiler")
  endif()
endif()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Prepare Generated Include Directory for Config Header
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Ensure the binary include folder exists so configure_file can write into it
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Auto-generate a config header with version macros
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/aerodynamics_physics_plugin_config.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/include/aerodynamics_physics_plugin_config.h"
  @ONLY
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# pkg-config file generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/aerodynamics_physics_plugin.pc.in"
  "${CMAKE_CURRENT_BINARY_DIR}/aerodynamics_physics_plugin.pc"
  @ONLY
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Collect C sources
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
file(GLOB_RECURSE PLUGIN_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Core library target (only after sources & config files are set up)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
add_library(${PROJECT_NAME} ${PLUGIN_SOURCES})
add_library(aerodynamics::physics_plugin ALIAS ${PROJECT_NAME})

if(MSVC)
  target_compile_options(${PROJECT_NAME} PRIVATE /W4 /WX)
else()
  target_compile_options(${PROJECT_NAME}
    PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -std=c11
    -Werror

    # Don't elevate this macOS-only warning into an error:
    -Wno-error=unused-command-line-argument)
endif()

# Public headers (install and build)
target_include_directories(${PROJECT_NAME}
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Optional C-API wrapper (src/c_api/*.c)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
file(GLOB_RECURSE C_API_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/c_api/*.c")

if(C_API_SOURCES)
  add_library(aerodynamics_c_api STATIC ${C_API_SOURCES})
  target_include_directories(aerodynamics_c_api PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )
  target_link_libraries(aerodynamics_c_api
    PUBLIC aerodynamics::physics_plugin
  )
  add_library(aerodynamics::c_api ALIAS aerodynamics_c_api)
endif()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Examples
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if(ENABLE_EXAMPLES)
  file(GLOB_RECURSE EXAMPLE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.c")

  foreach(example_file ${EXAMPLE_SRC})
    get_filename_component(example_name ${example_file} NAME_WE)
    add_executable(${example_name} ${example_file})
    target_link_libraries(${example_name} PRIVATE aerodynamics::physics_plugin)
  endforeach()
endif()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Unit Tests (CMocka + CTest)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if(ENABLE_TESTING)
  enable_testing()
  include(FetchContent)

  FetchContent_Declare(
    cmocka
    GIT_REPOSITORY https://gitlab.com/cmocka/cmocka.git
    GIT_TAG cmocka-1.1.8
  )
  FetchContent_MakeAvailable(cmocka)

  # Build one test executable per test_*.c
  file(GLOB TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/tests/aerodynamics/test_*.c")

  foreach(test_src IN LISTS TEST_FILES)
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_include_directories(${test_name}
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_BINARY_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/tests
      ${CMAKE_CURRENT_BINARY_DIR}/_deps/cmocka-src/inclde
    )
    target_link_libraries(${test_name}
      PRIVATE
      aerodynamics::physics_plugin
      cmocka
    )

    # Wrap on Linux only
    if(UNIX AND NOT APPLE AND NOT MSVC)
      target_link_options(${test_name} PRIVATE
        -Wl, --wrap=calloc
        -Wl, --wrap=malloc
      )
      target_compile_definitions(${test_name} PRIVATE MOCK_CALLOC)
    endif()

    add_test(NAME ${test_name} COMMAND ${test_name})
  endforeach()

  # ─── Whole-Pipeline Smoke Test ───────────────────────────────────────────────
  add_executable(test_full_pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/integration/test_full_pipeline.c
  )

  target_include_directories(test_full_pipeline
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include
  )

  target_link_libraries(test_full_pipeline
    PRIVATE
    aerodynamics::physics_plugin
  )

  # Register the smoke-test with CTest.
  add_test(NAME full_pipeline_c COMMAND test_full_pipeline)
endif()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Installation & Packaging
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
install(TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

if(TARGET aerodynamics_c_api)
  install(TARGETS aerodynamics_c_api
    EXPORT ${PROJECT_NAME}Targets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
  )
endif()

install(DIRECTORY include/ DESTINATION include)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/include/aerodynamics_physics_plugin_config.h"
  DESTINATION include
)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/aerodynamics_physics_plugin.pc"
  DESTINATION lib/pkgconfig
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION lib/cmake/${PROJECT_NAME}
)

install(
  EXPORT ${PROJECT_NAME}Targets
  FILE ${PROJECT_NAME}Config.cmake
  NAMESPACE aerodynamics::
  DESTINATION lib/cmake/${PROJECT_NAME}
)

include(CPack)
set(CPACK_PACKAGE_NAME "${PROJECT_NAME}")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_GENERATOR "TGZ;ZIP")

message(STATUS "Configuring ${PROJECT_NAME} v${PROJECT_VERSION}")
message(STATUS "  Shared libs   : ${BUILD_SHARED_LIBS}")
message(STATUS "  Coverage      : ${ENABLE_COVERAGE}")
message(STATUS "  Testing       : ${ENABLE_TESTING}")
message(STATUS "  Examples      : ${ENABLE_EXAMPLES}")
