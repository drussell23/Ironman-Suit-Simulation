#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "aerodynamics::aerodynamics_physics_plugin" for configuration ""
set_property(TARGET aerodynamics::aerodynamics_physics_plugin APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(aerodynamics::aerodynamics_physics_plugin PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libaerodynamics_physics_plugin.dylib"
  IMPORTED_SONAME_NOCONFIG "@rpath/libaerodynamics_physics_plugin.dylib"
  )

list(APPEND _cmake_import_check_targets aerodynamics::aerodynamics_physics_plugin )
list(APPEND _cmake_import_check_files_for_aerodynamics::aerodynamics_physics_plugin "${_IMPORT_PREFIX}/lib/libaerodynamics_physics_plugin.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
