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
include CMakeFiles/test_bindings.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/test_bindings.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/test_bindings.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_bindings.dir/flags.make

CMakeFiles/test_bindings.dir/codegen:
.PHONY : CMakeFiles/test_bindings.dir/codegen

CMakeFiles/test_bindings.dir/tests/aerodynamics/test_bindings.c.o: CMakeFiles/test_bindings.dir/flags.make
CMakeFiles/test_bindings.dir/tests/aerodynamics/test_bindings.c.o: /Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/tests/aerodynamics/test_bindings.c
CMakeFiles/test_bindings.dir/tests/aerodynamics/test_bindings.c.o: CMakeFiles/test_bindings.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/test_bindings.dir/tests/aerodynamics/test_bindings.c.o"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/test_bindings.dir/tests/aerodynamics/test_bindings.c.o -MF CMakeFiles/test_bindings.dir/tests/aerodynamics/test_bindings.c.o.d -o CMakeFiles/test_bindings.dir/tests/aerodynamics/test_bindings.c.o -c /Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/tests/aerodynamics/test_bindings.c

CMakeFiles/test_bindings.dir/tests/aerodynamics/test_bindings.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/test_bindings.dir/tests/aerodynamics/test_bindings.c.i"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/tests/aerodynamics/test_bindings.c > CMakeFiles/test_bindings.dir/tests/aerodynamics/test_bindings.c.i

CMakeFiles/test_bindings.dir/tests/aerodynamics/test_bindings.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/test_bindings.dir/tests/aerodynamics/test_bindings.c.s"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/tests/aerodynamics/test_bindings.c -o CMakeFiles/test_bindings.dir/tests/aerodynamics/test_bindings.c.s

# Object files for target test_bindings
test_bindings_OBJECTS = \
"CMakeFiles/test_bindings.dir/tests/aerodynamics/test_bindings.c.o"

# External object files for target test_bindings
test_bindings_EXTERNAL_OBJECTS =

test_bindings: CMakeFiles/test_bindings.dir/tests/aerodynamics/test_bindings.c.o
test_bindings: CMakeFiles/test_bindings.dir/build.make
test_bindings: libaerodynamics_physics_plugin.dylib
test_bindings: _deps/cmocka-build/src/libcmocka.0.8.0.dylib
test_bindings: CMakeFiles/test_bindings.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable test_bindings"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_bindings.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_bindings.dir/build: test_bindings
.PHONY : CMakeFiles/test_bindings.dir/build

CMakeFiles/test_bindings.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_bindings.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_bindings.dir/clean

CMakeFiles/test_bindings.dir/depend:
	cd /Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin /Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin /Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/build /Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/build /Users/derekjrussell/Documents/repos/IronMan/backend/aerodynamics/physics_plugin/build/CMakeFiles/test_bindings.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/test_bindings.dir/depend

