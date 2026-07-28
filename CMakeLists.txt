cmake_minimum_required(VERSION 3.10)
project(genshin_slam LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV
set(OpenCV_DIR D:/Path/opencv)
find_package(OpenCV REQUIRED)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# Source files
set(SLAM_SOURCES
    src/se3.cpp
    src/ekf_slam.cpp
    src/graph_slam.cpp
)

# Library
add_library(slam_core STATIC ${SLAM_SOURCES})
target_link_libraries(slam_core ${OpenCV_LIBS})
target_include_directories(slam_core PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Mock test executable
add_executable(mock_test src/mock_test.cpp)
target_link_libraries(mock_test slam_core ${OpenCV_LIBS})

# Graph test executable
add_executable(graph_test src/graph_test.cpp)
target_link_libraries(graph_test slam_core ${OpenCV_LIBS})