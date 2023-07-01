set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

add_library(bytetrack SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/bytetrack.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/BYTETracker.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kalmanFilter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lapjv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/STrack.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cpp
)
target_compile_features(bytetrack PRIVATE cxx_std_17)
target_include_directories(bytetrack PRIVATE
    PUBLIC ${PROJECT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(bytetrack PRIVATE
    ${OpenCV_LIBS}
    yoloxv3
)
