# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.31

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/build

# Include any dependencies generated for this target.
include _deps/cmocka-build/example/CMakeFiles/assert_macro_test.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include _deps/cmocka-build/example/CMakeFiles/assert_macro_test.dir/compiler_depend.make

# Include the progress variables for this target.
include _deps/cmocka-build/example/CMakeFiles/assert_macro_test.dir/progress.make

# Include the compile flags for this target's objects.
include _deps/cmocka-build/example/CMakeFiles/assert_macro_test.dir/flags.make

_deps/cmocka-build/example/CMakeFiles/assert_macro_test.dir/codegen:
.PHONY : _deps/cmocka-build/example/CMakeFiles/assert_macro_test.dir/codegen

_deps/cmocka-build/example/CMakeFiles/assert_macro_test.dir/assert_macro.c.o: _deps/cmocka-build/example/CMakeFiles/assert_macro_test.dir/flags.make
_deps/cmocka-build/example/CMakeFiles/assert_macro_test.dir/assert_macro.c.o: _deps/cmocka-src/example/assert_macro.c
_deps/cmocka-build/example/CMakeFiles/assert_macro_test.dir/assert_macro.c.o: _deps/cmocka-build/example/CMakeFiles/assert_macro_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object _deps/cmocka-build/example/CMakeFiles/assert_macro_test.dir/assert_macro.c.o"
	cd /Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/build/_deps/cmocka-build/example && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT _deps/cmocka-build/example/CMakeFiles/assert_macro_test.dir/assert_macro.c.o -MF CMakeFiles/assert_macro_test.dir/assert_macro.c.o.d -o CMakeFiles/assert_macro_test.dir/assert_macro.c.o -c /Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/build/_deps/cmocka-src/example/assert_macro.c

_deps/cmocka-build/example/CMakeFiles/assert_macro_test.dir/assert_macro.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/assert_macro_test.dir/assert_macro.c.i"
	cd /Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/build/_deps/cmocka-build/example && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/build/_deps/cmocka-src/example/assert_macro.c > CMakeFiles/assert_macro_test.dir/assert_macro.c.i

_deps/cmocka-build/example/CMakeFiles/assert_macro_test.dir/assert_macro.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/assert_macro_test.dir/assert_macro.c.s"
	cd /Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/build/_deps/cmocka-build/example && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/build/_deps/cmocka-src/example/assert_macro.c -o CMakeFiles/assert_macro_test.dir/assert_macro.c.s

_deps/cmocka-build/example/CMakeFiles/assert_macro_test.dir/assert_macro_test.c.o: _deps/cmocka-build/example/CMakeFiles/assert_macro_test.dir/flags.make
_deps/cmocka-build/example/CMakeFiles/assert_macro_test.dir/assert_macro_test.c.o: _deps/cmocka-src/example/assert_macro_test.c
_deps/cmocka-build/example/CMakeFiles/assert_macro_test.dir/assert_macro_test.c.o: _deps/cmocka-build/example/CMakeFiles/assert_macro_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object _deps/cmocka-build/example/CMakeFiles/assert_macro_test.dir/assert_macro_test.c.o"
	cd /Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/build/_deps/cmocka-build/example && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT _deps/cmocka-build/example/CMakeFiles/assert_macro_test.dir/assert_macro_test.c.o -MF CMakeFiles/assert_macro_test.dir/assert_macro_test.c.o.d -o CMakeFiles/assert_macro_test.dir/assert_macro_test.c.o -c /Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/build/_deps/cmocka-src/example/assert_macro_test.c

_deps/cmocka-build/example/CMakeFiles/assert_macro_test.dir/assert_macro_test.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/assert_macro_test.dir/assert_macro_test.c.i"
	cd /Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/build/_deps/cmocka-build/example && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/build/_deps/cmocka-src/example/assert_macro_test.c > CMakeFiles/assert_macro_test.dir/assert_macro_test.c.i

_deps/cmocka-build/example/CMakeFiles/assert_macro_test.dir/assert_macro_test.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/assert_macro_test.dir/assert_macro_test.c.s"
	cd /Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/build/_deps/cmocka-build/example && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/build/_deps/cmocka-src/example/assert_macro_test.c -o CMakeFiles/assert_macro_test.dir/assert_macro_test.c.s

# Object files for target assert_macro_test
assert_macro_test_OBJECTS = \
"CMakeFiles/assert_macro_test.dir/assert_macro.c.o" \
"CMakeFiles/assert_macro_test.dir/assert_macro_test.c.o"

# External object files for target assert_macro_test
assert_macro_test_EXTERNAL_OBJECTS =

_deps/cmocka-build/example/assert_macro_test: _deps/cmocka-build/example/CMakeFiles/assert_macro_test.dir/assert_macro.c.o
_deps/cmocka-build/example/assert_macro_test: _deps/cmocka-build/example/CMakeFiles/assert_macro_test.dir/assert_macro_test.c.o
_deps/cmocka-build/example/assert_macro_test: _deps/cmocka-build/example/CMakeFiles/assert_macro_test.dir/build.make
_deps/cmocka-build/example/assert_macro_test: _deps/cmocka-build/src/libcmocka.0.8.0.dylib
_deps/cmocka-build/example/assert_macro_test: _deps/cmocka-build/example/CMakeFiles/assert_macro_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking C executable assert_macro_test"
	cd /Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/build/_deps/cmocka-build/example && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/assert_macro_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
_deps/cmocka-build/example/CMakeFiles/assert_macro_test.dir/build: _deps/cmocka-build/example/assert_macro_test
.PHONY : _deps/cmocka-build/example/CMakeFiles/assert_macro_test.dir/build

_deps/cmocka-build/example/CMakeFiles/assert_macro_test.dir/clean:
	cd /Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/build/_deps/cmocka-build/example && $(CMAKE_COMMAND) -P CMakeFiles/assert_macro_test.dir/cmake_clean.cmake
.PHONY : _deps/cmocka-build/example/CMakeFiles/assert_macro_test.dir/clean

_deps/cmocka-build/example/CMakeFiles/assert_macro_test.dir/depend:
	cd /Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin /Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/build/_deps/cmocka-src/example /Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/build /Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/build/_deps/cmocka-build/example /Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/build/_deps/cmocka-build/example/CMakeFiles/assert_macro_test.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : _deps/cmocka-build/example/CMakeFiles/assert_macro_test.dir/depend

