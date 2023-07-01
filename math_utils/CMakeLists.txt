set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
file(GLOB _SRCS "*.cc" "*.cpp" "*.h")
add_library(math_utils OBJECT ${_SRCS})

target_compile_features(math_utils PRIVATE cxx_std_17)
target_include_directories(math_utils PRIVATE
    PUBLIC ${PROJECT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(math_utils PRIVATE
    common
)

