cmake_minimum_required(VERSION 3.16)
project(spinalflow_sim_builder_tests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Adjust these include/src dirs to your repo structure
include_directories(
    ${CMAKE_SOURCE_DIR}/include
)

# Core library sources
set(SF_CORE_SOURCES
    src/arch/filter_buffer.cpp
    src/arch/global_merger.cpp
    src/arch/input_spine_buffer.cpp
    src/arch/input_weight_provider.cpp
    src/arch/intermediate_fifo.cpp
    src/arch/min_finder_batch.cpp
    src/arch/output_queue.cpp
    src/arch/pe_array.cpp
    src/arch/smallest_ts_picker.cpp
    src/arch/driver/weight_lut.cpp
    src/arch/dram/fixed_stride_format.cpp
    src/arch/dram/input_spine_fetcher.cpp
    src/arch/dram/output_spine_writer.cpp
    src/arch/dram/packed_format.cpp
    src/arch/dram/stream_reader.cpp
    src/arch/dram/weight_loader.cpp
    src/core/clock.cpp
    src/core/event_queue.cpp
    src/core/logger.cpp
    src/core/stats.cpp
    src/runner/conv_runner.cpp
)

add_library(sf_core STATIC ${SF_CORE_SOURCES})

# If some of the above live in other dirs, add them; or if you already have a library,
# you can remove this target and link against your existing target instead.

# Smoke executable for ConvRunner end-to-end flow
add_executable(smoke_conv_layer tests/smoke/smoke_conv_layer.cpp)
target_link_libraries(smoke_conv_layer PRIVATE sf_core)

# Helpful flags
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(sf_core PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(smoke_conv_layer PRIVATE -Wall -Wextra -Wpedantic)
endif()
