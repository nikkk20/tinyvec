cmake_minimum_required(VERSION 3.20)
project(tinyvec LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(TINYVEC_BUILD_TESTS "Build tinyvec tests" ON)
option(TINYVEC_BUILD_BENCHMARKS "Build tinyvec benchmarks" ON)

add_library(tinyvec_core STATIC
    src/core/dispatch.cpp
    src/core/distance_scalar.cpp
    src/core/distance_avx2.cpp
    src/core/distance_avx512.cpp
    src/index/flat_index.cpp
)
target_include_directories(tinyvec_core PUBLIC include)

if (MSVC)
    set_source_files_properties(src/core/distance_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    set_source_files_properties(src/core/distance_avx512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
else()
    set_source_files_properties(src/core/distance_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
    set_source_files_properties(src/core/distance_avx512.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f;-mfma")
endif()

if (TINYVEC_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if (TINYVEC_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
