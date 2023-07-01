cmake_minimum_required(VERSION 3.10.2)
project(main LANGUAGES CXX VERSION 0.0.1)

# -fPIC：生成动态库，-fopenmp 开启多线程，-O3 对代码进行优化，
# -g 打印调试信息，-Wall 打印所有警告信息, pthread 支持多线程
# -pie 生成位置无关的可执行文件
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=c++17 -g -O3 -fopenmp -pthread -ldl -pie ")

# 不生成警告信息: debug阶段尽量不要使用
add_definitions(-w)

option(BUILD_TEST "Build all tests." ON)
option(BUILD_EXAMPLE "Build examples." ON)

find_package(Boost REQUIRED COMPONENTS thread filesystem program_options system)

link_directories(
    ${Boost_INCLUDE_DIRS}
)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else ()
    message(FATAL_ERROR "OpenMP Not Found!")
endif ()

#必须为绝对路径，如有多个文件使用空格隔开
set(EZCFG_STRUCT_HEADER_FILE ${CMAKE_CURRENT_SOURCE_DIR}/config/config.h)

add_subdirectory(third_party/backward-cpp)
add_subdirectory(third_party/easy_config)
add_subdirectory(common)
add_subdirectory(modules)

if(BUILD_TEST)
    add_subdirectory(examples)
endif()

find_package(OpenCV 3 REQUIRED)
link_directories(${OpenCV_LIBS})
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(main main.cpp)
target_link_libraries(main PRIVATE
    ${BACKWARD_ENABLE}
    ${OpenCV_LIBS}
    common
    yolopv2
    yoloxv3
    ezcfg::ezcfg
)
add_backward(${PROJECT_NAME})