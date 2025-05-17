# CMake generated Testfile for 
# Source directory: /Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin
# Build directory: /Users/derekjrussell/Documents/repos/IronMan/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(aerodynamics_unit_tests "/Users/derekjrussell/Documents/repos/IronMan/build/aerodynamics_physics_plugin_tests")
set_tests_properties(aerodynamics_unit_tests PROPERTIES  _BACKTRACE_TRIPLES "/Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/CMakeLists.txt;155;add_test;/Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/CMakeLists.txt;0;")
subdirs("_deps/cmocka-build")
