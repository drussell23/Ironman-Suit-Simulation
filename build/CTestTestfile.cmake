# CMake generated Testfile for 
# Source directory: /Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin
# Build directory: /Users/derekjrussell/Documents/repos/IronMan/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_actuator "/Users/derekjrussell/Documents/repos/IronMan/build/test_actuator")
set_tests_properties(test_actuator PROPERTIES  _BACKTRACE_TRIPLES "/Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/CMakeLists.txt;182;add_test;/Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/CMakeLists.txt;0;")
add_test(test_bindings "/Users/derekjrussell/Documents/repos/IronMan/build/test_bindings")
set_tests_properties(test_bindings PROPERTIES  _BACKTRACE_TRIPLES "/Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/CMakeLists.txt;182;add_test;/Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/CMakeLists.txt;0;")
add_test(test_flow_state "/Users/derekjrussell/Documents/repos/IronMan/build/test_flow_state")
set_tests_properties(test_flow_state PROPERTIES  _BACKTRACE_TRIPLES "/Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/CMakeLists.txt;182;add_test;/Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/CMakeLists.txt;0;")
add_test(test_mesh "/Users/derekjrussell/Documents/repos/IronMan/build/test_mesh")
set_tests_properties(test_mesh PROPERTIES  _BACKTRACE_TRIPLES "/Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/CMakeLists.txt;182;add_test;/Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/CMakeLists.txt;0;")
add_test(test_solver "/Users/derekjrussell/Documents/repos/IronMan/build/test_solver")
set_tests_properties(test_solver PROPERTIES  _BACKTRACE_TRIPLES "/Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/CMakeLists.txt;182;add_test;/Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/CMakeLists.txt;0;")
add_test(test_turbulence_model "/Users/derekjrussell/Documents/repos/IronMan/build/test_turbulence_model")
set_tests_properties(test_turbulence_model PROPERTIES  _BACKTRACE_TRIPLES "/Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/CMakeLists.txt;182;add_test;/Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/CMakeLists.txt;0;")
add_test(full_pipeline_c "/Users/derekjrussell/Documents/repos/IronMan/build/test_full_pipeline")
set_tests_properties(full_pipeline_c PROPERTIES  _BACKTRACE_TRIPLES "/Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/CMakeLists.txt;202;add_test;/Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/CMakeLists.txt;0;")
subdirs("_deps/cmocka-build")
