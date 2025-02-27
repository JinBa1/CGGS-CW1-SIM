# Set the project name
cmake_minimum_required(VERSION 3.20)
project(CourseWork1Simulation)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Find OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found! Enabling parallel processing")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    add_compile_definitions(HAVE_OPENMP)
else()
    message(WARNING "OpenMP not found. Parallel processing will be disabled.")
endif()

# Specify the directories for Eigen and Polyscope
set(EIGEN3_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/../external/eigen)
set(POLYSCOPE_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/../external/polyscope/include)
set(CW0_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)

add_subdirectory("${CMAKE_SOURCE_DIR}/../external/polyscope" "polyscope")

add_library(common_includes INTERFACE)
target_include_directories(common_includes INTERFACE
    ${EIGEN3_INCLUDE_DIR}
    ${POLYSCOPE_INCLUDE_DIR}
    ${CW0_INCLUDE_DIR}
)

# Link OpenMP to common_includes if found
if(OpenMP_CXX_FOUND)
    target_link_libraries(common_includes INTERFACE OpenMP::OpenMP_CXX)
endif()

add_compile_definitions(DATA_PATH="${PROJECT_SOURCE_DIR}/../data")
set(TUTORIAL_DATA_PATH ${PROJECT_SOURCE_DIR}/../data CACHE PATH "location of tutorial data resources")
target_compile_definitions(common_includes INTERFACE "-DTUTORIAL_DATA_PATH=\"${TUTORIAL_DATA_PATH}\"")

# The subprojects
add_subdirectory(Section12)
add_subdirectory(Grading1)
add_subdirectory(Grading2)



