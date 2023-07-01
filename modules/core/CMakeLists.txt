set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

file(GLOB _SRCS "*.cc" "*.cpp" "*.hpp")
add_library(cloud_core SHARED ${_SRCS})

find_package(PCL 1.7 REQUIRED)
add_definitions(${PCL_DEFINITIONS})

target_compile_features(cloud_core PRIVATE cxx_std_17)
target_include_directories(cloud_core PRIVATE
    PUBLIC ${PROJECT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PCL_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/third_party/yaml/include
)
target_link_libraries(cloud_core PRIVATE
    common
    ${PROJECT_SOURCE_DIR}/third_party/yaml/lib/libyaml-cpp.so.0.7.0
    ${PCL_LIBRARY_DIRS}
)

