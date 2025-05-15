cmake_minimum_required(VERSION 3.18)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Project Setup (C-only)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
project(aerodynamics_physics_plugin
    VERSION     1.0.0
    DESCRIPTION "Aerodynamics physics plugin (CFD solver + turbulence models)"
    LANGUAGES   C
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# User Options
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
option(BUILD_SHARED_LIBS   "Build as a shared library (.so/.dll)" ON)
option(ENABLE_TESTING      "Enable building and running unit tests" ON)
option(ENABLE_EXAMPLES     "Enable building example executables" OFF)
option(ENABLE_COVERAGE     "Enable code-coverage instrumentation" OFF)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Compiler Standards & Flags
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
set(CMAKE_C_STANDARD            11)
set(CMAKE_C_STANDARD_REQUIRED   TRUE)
set(CMAKE_C_EXTENSIONS          OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MSVC)
  add_compile_options(/W4 /WX)
else()
  add_compile_options(-Wall -Wextra -Wpedantic -std=c11 -Werror)
endif()

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
    GIT_TAG        v1.1.5
  )
  FetchContent_MakeAvailable(cmocka)

  file(GLOB_RECURSE TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.c")
  add_executable(${PROJECT_NAME}_tests ${TEST_SOURCES})
  target_link_libraries(${PROJECT_NAME}_tests
    PRIVATE
      aerodynamics::physics_plugin
      cmocka
  )
  add_test(NAME unit_tests COMMAND ${PROJECT_NAME}_tests)
endif()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Installation & Packaging
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
install(TARGETS ${PROJECT_NAME}
  EXPORT   ${PROJECT_NAME}Targets
  ARCHIVE  DESTINATION lib
  LIBRARY  DESTINATION lib
  RUNTIME  DESTINATION bin
)

if(TARGET aerodynamics_c_api)
  install(TARGETS aerodynamics_c_api
    EXPORT   ${PROJECT_NAME}Targets
    ARCHIVE  DESTINATION lib
    LIBRARY  DESTINATION lib
    RUNTIME  DESTINATION bin
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
  VERSION       ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION lib/cmake/${PROJECT_NAME}
)

install(
  EXPORT  ${PROJECT_NAME}Targets
  FILE    ${PROJECT_NAME}Config.cmake
  NAMESPACE aerodynamics::
  DESTINATION lib/cmake/${PROJECT_NAME}
)

include(CPack)
set(CPACK_PACKAGE_NAME        "${PROJECT_NAME}")
set(CPACK_PACKAGE_VERSION     "${PROJECT_VERSION}")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_GENERATOR           "TGZ;ZIP")

message(STATUS "Configuring ${PROJECT_NAME} v${PROJECT_VERSION}")
message(STATUS "  Shared libs   : ${BUILD_SHARED_LIBS}")
message(STATUS "  Coverage      : ${ENABLE_COVERAGE}")
message(STATUS "  Testing       : ${ENABLE_TESTING}")
message(STATUS "  Examples      : ${ENABLE_EXAMPLES}")
