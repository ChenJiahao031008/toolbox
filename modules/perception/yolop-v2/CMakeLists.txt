set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pie -fPIE -fPIC -Wall")

add_library(yolopv2 OBJECT yolop.cc)
# message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(yolopv2 PRIVATE cxx_std_17)
target_include_directories(yolopv2 PRIVATE
    PUBLIC ${PROJECT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/ncnn
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(yolopv2 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/libncnn.a
    ${OpenCV_LIBS}
)
