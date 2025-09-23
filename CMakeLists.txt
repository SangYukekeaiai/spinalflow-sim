cmake_minimum_required(VERSION 3.16)
project(spinalflow_sim_builder_tests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Adjust these include/src dirs to your repo structure
include_directories(
    ${CMAKE_SOURCE_DIR}/include
)

# Core library (adjust the source list to match your repo)
add_library(sf_core STATIC
    src/arch/builder.cpp
    src/arch/input_spine_buffer.cpp
    src/arch/intermediate_fifo.cpp
    src/arch/min_finder_batch.cpp
    src/arch/smallest_ts_picker.cpp
    src/arch/global_merger.cpp
    # Add your other sources if needed:
    src/arch/filter_buffer.cpp
    src/arch/pe.cpp
    src/arch/output_queue.cpp
    src/arch/driver/batch_spine_map.cpp
    src/arch/driver/weight_lut.cpp
)

# If some of the above live in other dirs, add them; or if you already have a library,
# you can remove this target and link against your existing target instead.

# Test executable
add_executable(test_builder_min
    tests/test_builder_min.cpp
)

target_link_libraries(test_builder_min PRIVATE sf_core)

# Helpful flags
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(sf_core PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(test_builder_min PRIVATE -Wall -Wextra -Wpedantic)
endif()
