cmake_minimum_required(VERSION 3.10)
project(genshin_slam LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================
# Third-party libraries (MinGW-built, manually linked)
# ============================================================
set(SLAM_DEPS_DIR D:/Path)

# Eigen3 (header-only)
set(EIGEN_INCLUDE_DIR ${SLAM_DEPS_DIR}/Eigen3/include/eigen3)

# Sophus (header-only, requires SOPHUS_USE_BASIC_LOGGING when built without fmt)
set(SOPHUS_INCLUDE_DIR ${SLAM_DEPS_DIR}/sophus/include)

# Ceres Solver 2.2 (static lib; cmake config is broken, link manually)
set(CERES_INCLUDE_DIR ${SLAM_DEPS_DIR}/Ceres/include)
set(CERES_LIB ${SLAM_DEPS_DIR}/Ceres/lib/libceres.a)

# glog (Ceres dependency; consumers must define GLOG_USE_GLOG_EXPORT)
set(GLOG_INCLUDE_DIR ${SLAM_DEPS_DIR}/glog/include)
set(GLOG_LIB ${SLAM_DEPS_DIR}/glog/lib/libglog.dll.a)

# SuiteSparse CXSparse (Ceres sparse backend)
set(CXSPARSE_INCLUDE_DIR ${SLAM_DEPS_DIR}/SuiteSparse/include/CXSparse)
set(CXSPARSE_LIB ${SLAM_DEPS_DIR}/SuiteSparse/lib/CXSparse/libcxsparse.dll.a)

# ============================================================
# Core SLAM library (no OpenCV dependency)
# ============================================================
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${EIGEN_INCLUDE_DIR}
    ${SOPHUS_INCLUDE_DIR}
    ${CERES_INCLUDE_DIR}
    ${GLOG_INCLUDE_DIR}
    ${CXSPARSE_INCLUDE_DIR}
)

set(SLAM_SOURCES
    src/ekf_slam.cpp
    src/graph_slam.cpp
)

add_library(slam_core STATIC ${SLAM_SOURCES})
target_include_directories(slam_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${EIGEN_INCLUDE_DIR}
    ${SOPHUS_INCLUDE_DIR}
)
target_compile_definitions(slam_core PUBLIC
    SOPHUS_USE_BASIC_LOGGING=1
    GLOG_USE_GLOG_EXPORT=1
)

target_link_libraries(slam_core
    ${CERES_LIB}
    ${GLOG_LIB}
    ${CXSPARSE_LIB}
)

# ============================================================
# Mock tests
# ============================================================
add_executable(mock_test src/mock_test.cpp)
target_link_libraries(mock_test slam_core)
target_compile_definitions(mock_test PRIVATE
    SOPHUS_USE_BASIC_LOGGING=1 GLOG_USE_GLOG_EXPORT=1)

add_executable(graph_test src/graph_test.cpp)
target_link_libraries(graph_test slam_core)
target_compile_definitions(graph_test PRIVATE
    SOPHUS_USE_BASIC_LOGGING=1 GLOG_USE_GLOG_EXPORT=1)

# ============================================================
# VO frontend + test (needs OpenCV for video I/O / features)
# ============================================================
set(OpenCV_DIR ${SLAM_DEPS_DIR}/opencv)
find_package(OpenCV REQUIRED)

add_executable(vo_test src/vo_test.cpp src/vo_frontend.cpp)
target_include_directories(vo_test PRIVATE ${OpenCV_INCLUDE_DIRS})
target_compile_definitions(vo_test PRIVATE
    SOPHUS_USE_BASIC_LOGGING=1 GLOG_USE_GLOG_EXPORT=1)
target_link_libraries(vo_test slam_core ${OpenCV_LIBS})