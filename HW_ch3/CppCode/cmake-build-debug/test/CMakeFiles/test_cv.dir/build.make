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
include test/CMakeFiles/test_cv.dir/depend.make

# Include the progress variables for this target.
include test/CMakeFiles/test_cv.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/test_cv.dir/flags.make

test/CMakeFiles/test_cv.dir/test_cv.cpp.o: test/CMakeFiles/test_cv.dir/flags.make
test/CMakeFiles/test_cv.dir/test_cv.cpp.o: ../test/test_cv.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jiayao/Desktop/cv-hw/HW_ch3/CppCode/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/CMakeFiles/test_cv.dir/test_cv.cpp.o"
	cd /home/jiayao/Desktop/cv-hw/HW_ch3/CppCode/cmake-build-debug/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_cv.dir/test_cv.cpp.o -c /home/jiayao/Desktop/cv-hw/HW_ch3/CppCode/test/test_cv.cpp

test/CMakeFiles/test_cv.dir/test_cv.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_cv.dir/test_cv.cpp.i"
	cd /home/jiayao/Desktop/cv-hw/HW_ch3/CppCode/cmake-build-debug/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jiayao/Desktop/cv-hw/HW_ch3/CppCode/test/test_cv.cpp > CMakeFiles/test_cv.dir/test_cv.cpp.i

test/CMakeFiles/test_cv.dir/test_cv.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_cv.dir/test_cv.cpp.s"
	cd /home/jiayao/Desktop/cv-hw/HW_ch3/CppCode/cmake-build-debug/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jiayao/Desktop/cv-hw/HW_ch3/CppCode/test/test_cv.cpp -o CMakeFiles/test_cv.dir/test_cv.cpp.s

# Object files for target test_cv
test_cv_OBJECTS = \
"CMakeFiles/test_cv.dir/test_cv.cpp.o"

# External object files for target test_cv
test_cv_EXTERNAL_OBJECTS =

test/test_cv: test/CMakeFiles/test_cv.dir/test_cv.cpp.o
test/test_cv: test/CMakeFiles/test_cv.dir/build.make
test/test_cv: src/liblvlset.a
test/test_cv: src/libheightmap.a
test/test_cv: utils/libutils.a
test/test_cv: src/liblvlset.a
test/test_cv: src/libheightmap.a
test/test_cv: utils/libutils.a
test/test_cv: src/libgdc_base.a
test/test_cv: /usr/local/lib/libopencv_stitching.so.3.4.2
test/test_cv: /usr/local/lib/libopencv_superres.so.3.4.2
test/test_cv: /usr/local/lib/libopencv_videostab.so.3.4.2
test/test_cv: /usr/local/lib/libopencv_aruco.so.3.4.2
test/test_cv: /usr/local/lib/libopencv_bgsegm.so.3.4.2
test/test_cv: /usr/local/lib/libopencv_bioinspired.so.3.4.2
test/test_cv: /usr/local/lib/libopencv_ccalib.so.3.4.2
test/test_cv: /usr/local/lib/libopencv_dnn_objdetect.so.3.4.2
test/test_cv: /usr/local/lib/libopencv_dpm.so.3.4.2
test/test_cv: /usr/local/lib/libopencv_face.so.3.4.2
test/test_cv: /usr/local/lib/libopencv_photo.so.3.4.2
test/test_cv: /usr/local/lib/libopencv_freetype.so.3.4.2
test/test_cv: /usr/local/lib/libopencv_fuzzy.so.3.4.2
test/test_cv: /usr/local/lib/libopencv_hfs.so.3.4.2
test/test_cv: /usr/local/lib/libopencv_img_hash.so.3.4.2
test/test_cv: /usr/local/lib/libopencv_line_descriptor.so.3.4.2
test/test_cv: /usr/local/lib/libopencv_optflow.so.3.4.2
test/test_cv: /usr/local/lib/libopencv_reg.so.3.4.2
test/test_cv: /usr/local/lib/libopencv_rgbd.so.3.4.2
test/test_cv: /usr/local/lib/libopencv_saliency.so.3.4.2
test/test_cv: /usr/local/lib/libopencv_sfm.so.3.4.2
test/test_cv: /usr/local/lib/libopencv_stereo.so.3.4.2
test/test_cv: /usr/local/lib/libopencv_structured_light.so.3.4.2
test/test_cv: /usr/local/lib/libopencv_phase_unwrapping.so.3.4.2
test/test_cv: /usr/local/lib/libopencv_surface_matching.so.3.4.2
test/test_cv: /usr/local/lib/libopencv_tracking.so.3.4.2
test/test_cv: /usr/local/lib/libopencv_datasets.so.3.4.2
test/test_cv: /usr/local/lib/libopencv_plot.so.3.4.2
test/test_cv: /usr/local/lib/libopencv_text.so.3.4.2
test/test_cv: /usr/local/lib/libopencv_dnn.so.3.4.2
test/test_cv: /usr/local/lib/libopencv_xfeatures2d.so.3.4.2
test/test_cv: /usr/local/lib/libopencv_ml.so.3.4.2
test/test_cv: /usr/local/lib/libopencv_shape.so.3.4.2
test/test_cv: /usr/local/lib/libopencv_video.so.3.4.2
test/test_cv: /usr/local/lib/libopencv_ximgproc.so.3.4.2
test/test_cv: /usr/local/lib/libopencv_calib3d.so.3.4.2
test/test_cv: /usr/local/lib/libopencv_features2d.so.3.4.2
test/test_cv: /usr/local/lib/libopencv_flann.so.3.4.2
test/test_cv: /usr/local/lib/libopencv_highgui.so.3.4.2
test/test_cv: /usr/local/lib/libopencv_videoio.so.3.4.2
test/test_cv: /usr/local/lib/libopencv_xobjdetect.so.3.4.2
test/test_cv: /usr/local/lib/libopencv_imgcodecs.so.3.4.2
test/test_cv: /usr/local/lib/libopencv_objdetect.so.3.4.2
test/test_cv: /usr/local/lib/libopencv_xphoto.so.3.4.2
test/test_cv: /usr/local/lib/libopencv_imgproc.so.3.4.2
test/test_cv: /usr/local/lib/libopencv_core.so.3.4.2
test/test_cv: test/CMakeFiles/test_cv.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jiayao/Desktop/cv-hw/HW_ch3/CppCode/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test_cv"
	cd /home/jiayao/Desktop/cv-hw/HW_ch3/CppCode/cmake-build-debug/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_cv.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/test_cv.dir/build: test/test_cv

.PHONY : test/CMakeFiles/test_cv.dir/build

test/CMakeFiles/test_cv.dir/clean:
	cd /home/jiayao/Desktop/cv-hw/HW_ch3/CppCode/cmake-build-debug/test && $(CMAKE_COMMAND) -P CMakeFiles/test_cv.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/test_cv.dir/clean

test/CMakeFiles/test_cv.dir/depend:
	cd /home/jiayao/Desktop/cv-hw/HW_ch3/CppCode/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jiayao/Desktop/cv-hw/HW_ch3/CppCode /home/jiayao/Desktop/cv-hw/HW_ch3/CppCode/test /home/jiayao/Desktop/cv-hw/HW_ch3/CppCode/cmake-build-debug /home/jiayao/Desktop/cv-hw/HW_ch3/CppCode/cmake-build-debug/test /home/jiayao/Desktop/cv-hw/HW_ch3/CppCode/cmake-build-debug/test/CMakeFiles/test_cv.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/test_cv.dir/depend

