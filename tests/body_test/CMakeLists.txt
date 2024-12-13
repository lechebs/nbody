cmake_minimum_required(VERSION 3.30)
project(nbody_body_test)

# C++ Standard
set(CMAKE_CXX_STANDARD 20)

# Include header files
include_directories("../../include/")

# Include source files
file(GLOB SOURCES "src/*.cpp")

# Main file name
set(MAIN_FILE "body_test.cpp")

# Target file name
set(TARGET_FILE "body_test")

add_executable("${TARGET_FILE}" "${MAIN_FILE}" ${SOURCES})