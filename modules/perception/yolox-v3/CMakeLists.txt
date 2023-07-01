set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

find_package(CUDA REQUIRED)
link_directories(/usr/local/cuda-11.1/lib64)
link_directories(/usr/local/cuda/lib64/)

add_library(yoloxv3 SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/yolox.cc
)
target_compile_features(yoloxv3 PRIVATE cxx_std_17)
target_include_directories(yoloxv3 PRIVATE
    PUBLIC ${PROJECT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    /usr/local/cuda-11.1/include/
    /usr/local/cuda/include/
)
target_link_libraries(yoloxv3 PRIVATE
    ${OpenCV_LIBS}
    nvinfer 
    cudart 
    nvonnxparser 
    nvparsers
)
