cmake_minimum_required(VERSION 3.15)
project(spinalflow-sim LANGUAGES CXX)

# ---------------- Compiler setup ----------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ---------------- Include dirs ----------------
# Headers under include/, e.g. include/arch/*.hpp and include/common/*.hpp
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---------------- Arch sources ----------------
set(SF_ARCH_SOURCES
    src/arch/pe.cpp
    src/arch/input_spine_buffer.cpp
    src/arch/min_finder_batch.cpp
    src/arch/filter_buffer.cpp
    src/arch/intermediate_fifo.cpp
    src/arch/global_merger.cpp
)

# ---------------- Driver sources ----------------
set(SF_DRIVER_SOURCES
    src/arch/driver/batch_spine_map.cpp
    src/arch/driver/weight_lut.cpp
)

# ---------------- Libraries ----------------
# Arch + Pipeline combined into one library for reuse
add_library(sf_arch STATIC
    ${SF_ARCH_SOURCES}
)

target_include_directories(sf_arch PUBLIC ${PROJECT_SOURCE_DIR}/include)

# Driver library (BatchSpineMap + WeightLUT)
add_library(sf_arch_driver STATIC
    ${SF_DRIVER_SOURCES}
)
target_include_directories(sf_arch_driver PUBLIC ${PROJECT_SOURCE_DIR}/include)

# ---------------- Example executables ----------------
add_executable(test_batch_mapping
    tests/driver/test_batch_mapping.cpp
)
target_include_directories(test_batch_mapping PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_batch_mapping PRIVATE sf_arch_driver)

# ---------------- Unit tests (without GTest) ----------------
# Simple unit test for WeightLUT
add_executable(test_weight_lut
    tests/unit/test_weight_lut.cpp
    ${SF_DRIVER_SOURCES}
)
target_include_directories(test_weight_lut PUBLIC ${PROJECT_SOURCE_DIR}/include)

# ---------------- (Optional) GTest integration ----------------
# Uncomment if you want GoogleTest instead of assert-based tests:
# find_package(GTest REQUIRED)
# add_executable(test_weight_lut_gtest
#     tests/driver/test_weight_lut_gtest.cpp
#     ${SF_DRIVER_SOURCES}
# )
# target_link_libraries(test_weight_lut_gtest PRIVATE sf_arch_driver GTest::gtest_main)
