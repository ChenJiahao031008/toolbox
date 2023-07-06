cmake_minimum_required(VERSION 3.10.2)
project(cloud_processing_example LANGUAGES CXX VERSION 0.0.1)

set(CMAKE_BUILD_TYPE Release)
# -fPIC：生成动态库，-fopenmp 开启多线程，-O3 对代码进行优化，-g 打印调试信息，-Wall 打印所有警告信息, pthread 支持多线程
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=c++17 -g -O3 -fopenmp -pthread")

find_package(PCL 1.7 REQUIRED)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../bin)
# add_executable(segment_dyna segment_dyna.cpp)
# target_include_directories(segment_dyna BEFORE PRIVATE
# ${PROJECT_SOURCE_DIR}/../
# ${PCL_INCLUDE_DIRS})
# target_compile_features(segment_dyna PRIVATE cxx_std_17)
# target_link_libraries(segment_dyna PRIVATE
#     common
#     ${BACKWARD_ENABLE}
#     ${PCL_LIBRARIES}
#     cloud_core
# )
# add_backward(segment_dyna)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../bin)

## segment_ground
add_executable(segment_ground 
    segment_ground.cpp)
target_include_directories(segment_ground BEFORE PRIVATE
    ${PROJECT_SOURCE_DIR}/../
    ${PROJECT_SOURCE_DIR}/../modules/pcl_extension/segment/
    ${PROJECT_SOURCE_DIR}/../third_party/yaml/include/
    ${PCL_INCLUDE_DIRS}
)
target_compile_features(segment_ground PRIVATE cxx_std_17)
target_link_libraries(segment_ground PRIVATE
    ${PROJECT_SOURCE_DIR}/../third_party/yaml/lib/libyaml-cpp.so.0.7.0
    ${BACKWARD_ENABLE}
    ${PCL_LIBRARIES}
    common
    segment
)
add_backward(segment_ground)


## segment_plane
add_executable(segment_plane segment_plane.cpp)
target_include_directories(segment_plane BEFORE PRIVATE
    ${PROJECT_SOURCE_DIR}/../
    ${PROJECT_SOURCE_DIR}/../modules/core/
    ${PCL_INCLUDE_DIRS})
target_compile_features(segment_plane PRIVATE cxx_std_17)
target_link_libraries(segment_plane PRIVATE
    common
    segment
    ${PROJECT_SOURCE_DIR}/../third_party/yaml/lib/libyaml-cpp.so.0.7.0
    ${BACKWARD_ENABLE}
    ${PCL_LIBRARIES}
)
add_backward(segment_plane)
