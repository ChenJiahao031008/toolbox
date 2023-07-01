set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

file(GLOB _SRCS "*.cc" "*.cpp" "*.h" "*.hpp")
add_library(common OBJECT ${_SRCS})

target_compile_features(common PRIVATE cxx_std_17)
target_include_directories(common PRIVATE
    PUBLIC ${PROJECT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# message(STATUS "Boost_LIBRARIES: ${Boost_LIBRARIES}")

target_link_libraries(common PRIVATE
    ${Boost_LIBRARIES}
    /usr/lib/x86_64-linux-gnu/libdl.so
    glog
    gflags
)

