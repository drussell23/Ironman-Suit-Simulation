# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/build/_deps/cmocka-src")
  file(MAKE_DIRECTORY "/Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/build/_deps/cmocka-src")
endif()
file(MAKE_DIRECTORY
  "/Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/build/_deps/cmocka-build"
  "/Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/build/_deps/cmocka-subbuild/cmocka-populate-prefix"
  "/Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/build/_deps/cmocka-subbuild/cmocka-populate-prefix/tmp"
  "/Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/build/_deps/cmocka-subbuild/cmocka-populate-prefix/src/cmocka-populate-stamp"
  "/Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/build/_deps/cmocka-subbuild/cmocka-populate-prefix/src"
  "/Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/build/_deps/cmocka-subbuild/cmocka-populate-prefix/src/cmocka-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/build/_deps/cmocka-subbuild/cmocka-populate-prefix/src/cmocka-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/build/_deps/cmocka-subbuild/cmocka-populate-prefix/src/cmocka-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
