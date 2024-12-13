cmake_minimum_required(VERSION 3.30)
project(nbody)

# C++ Standard
set(CMAKE_CXX_STANDARD 20)

# Include header files
include_directories(include)

# Include source files
file(GLOB SOURCES "src/*.cpp")

# Main file name
set(MAIN_FILE "main.cpp")

# Target file name
set(TARGET_FILE "nbody")

add_executable("${TARGET_FILE}" "${MAIN_FILE}" ${SOURCES})