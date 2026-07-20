add_executable(bench_distance bench_distance.cpp)
target_link_libraries(bench_distance PRIVATE tinyvec_core)
target_include_directories(bench_distance PRIVATE ${CMAKE_SOURCE_DIR}/src/core)
