# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.12

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /snap/clion/2018/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /snap/clion/2018/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jiayao/Desktop/cv-hw/HW_ch3/CppCode

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jiayao/Desktop/cv-hw/HW_ch3/CppCode/cmake-build-debug

# Include any dependencies generated for this target.
include src/CMakeFiles/gdc_base.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/gdc_base.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/gdc_base.dir/flags.make

src/CMakeFiles/gdc_base.dir/gradient_descent_base.cpp.o: src/CMakeFiles/gdc_base.dir/flags.make
src/CMakeFiles/gdc_base.dir/gradient_descent_base.cpp.o: ../src/gradient_descent_base.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jiayao/Desktop/cv-hw/HW_ch3/CppCode/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/gdc_base.dir/gradient_descent_base.cpp.o"
	cd /home/jiayao/Desktop/cv-hw/HW_ch3/CppCode/cmake-build-debug/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/gdc_base.dir/gradient_descent_base.cpp.o -c /home/jiayao/Desktop/cv-hw/HW_ch3/CppCode/src/gradient_descent_base.cpp

src/CMakeFiles/gdc_base.dir/gradient_descent_base.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/gdc_base.dir/gradient_descent_base.cpp.i"
	cd /home/jiayao/Desktop/cv-hw/HW_ch3/CppCode/cmake-build-debug/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jiayao/Desktop/cv-hw/HW_ch3/CppCode/src/gradient_descent_base.cpp > CMakeFiles/gdc_base.dir/gradient_descent_base.cpp.i

src/CMakeFiles/gdc_base.dir/gradient_descent_base.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/gdc_base.dir/gradient_descent_base.cpp.s"
	cd /home/jiayao/Desktop/cv-hw/HW_ch3/CppCode/cmake-build-debug/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jiayao/Desktop/cv-hw/HW_ch3/CppCode/src/gradient_descent_base.cpp -o CMakeFiles/gdc_base.dir/gradient_descent_base.cpp.s

# Object files for target gdc_base
gdc_base_OBJECTS = \
"CMakeFiles/gdc_base.dir/gradient_descent_base.cpp.o"

# External object files for target gdc_base
gdc_base_EXTERNAL_OBJECTS =

src/libgdc_base.a: src/CMakeFiles/gdc_base.dir/gradient_descent_base.cpp.o
src/libgdc_base.a: src/CMakeFiles/gdc_base.dir/build.make
src/libgdc_base.a: src/CMakeFiles/gdc_base.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jiayao/Desktop/cv-hw/HW_ch3/CppCode/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libgdc_base.a"
	cd /home/jiayao/Desktop/cv-hw/HW_ch3/CppCode/cmake-build-debug/src && $(CMAKE_COMMAND) -P CMakeFiles/gdc_base.dir/cmake_clean_target.cmake
	cd /home/jiayao/Desktop/cv-hw/HW_ch3/CppCode/cmake-build-debug/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/gdc_base.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/gdc_base.dir/build: src/libgdc_base.a

.PHONY : src/CMakeFiles/gdc_base.dir/build

src/CMakeFiles/gdc_base.dir/clean:
	cd /home/jiayao/Desktop/cv-hw/HW_ch3/CppCode/cmake-build-debug/src && $(CMAKE_COMMAND) -P CMakeFiles/gdc_base.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/gdc_base.dir/clean

src/CMakeFiles/gdc_base.dir/depend:
	cd /home/jiayao/Desktop/cv-hw/HW_ch3/CppCode/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jiayao/Desktop/cv-hw/HW_ch3/CppCode /home/jiayao/Desktop/cv-hw/HW_ch3/CppCode/src /home/jiayao/Desktop/cv-hw/HW_ch3/CppCode/cmake-build-debug /home/jiayao/Desktop/cv-hw/HW_ch3/CppCode/cmake-build-debug/src /home/jiayao/Desktop/cv-hw/HW_ch3/CppCode/cmake-build-debug/src/CMakeFiles/gdc_base.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/gdc_base.dir/depend

