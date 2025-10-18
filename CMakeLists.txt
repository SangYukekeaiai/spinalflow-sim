# All comments are in English.
cmake_minimum_required(VERSION 3.16)

# ---- Project ----
project(SpinalFlowSim
  VERSION 0.1.0
  DESCRIPTION "SpinalFlow SNN accelerator simulator"
  LANGUAGES CXX)

# ---- Options ----
option(SFS_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)

# ---- C++ standard ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- Build type default ----
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)
endif()

# ---- Output folders (optional) ----
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# ---- Sources ----
# Recursively collect all .cpp under src/
file(GLOB_RECURSE SFS_ALL_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

# Sanity: require main.cpp to exist
set(SFS_MAIN "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
if(NOT EXISTS "${SFS_MAIN}")
  message(FATAL_ERROR "src/main.cpp not found. Please ensure entry point exists.")
endif()

# Remove main.cpp from the core sources
set(SFS_CORE_SRC ${SFS_ALL_SRC})
list(REMOVE_ITEM SFS_CORE_SRC "${SFS_MAIN}")

# ---- Library: core (everything except main.cpp) ----
add_library(sfs_core STATIC ${SFS_CORE_SRC})

# Public headers live under include/
target_include_directories(sfs_core
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# ---- Warnings setup (applied to targets) ----
function(sfs_apply_warnings tgt)
  if(MSVC)
    target_compile_options(${tgt} PRIVATE /W4 /permissive- /EHsc)
    if(SFS_WARNINGS_AS_ERRORS)
      target_compile_options(${tgt} PRIVATE /WX)
    endif()
  else()
    target_compile_options(${tgt} PRIVATE -Wall -Wextra -Wpedantic)
    if(SFS_WARNINGS_AS_ERRORS)
      target_compile_options(${tgt} PRIVATE -Werror)
    endif()
  endif()

  if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT MSVC)
    target_compile_options(${tgt} PRIVATE -O0 -g)
  endif()
endfunction()

sfs_apply_warnings(sfs_core)

# ---- Executable: spinalflow-sim (main app) ----
add_executable(spinalflow-sim "${SFS_MAIN}")
target_link_libraries(spinalflow-sim PRIVATE sfs_core)
# No extra include dirs needed; sfs_core already exposes include/

sfs_apply_warnings(spinalflow-sim)

# ---- Executable: unit-test / validator for SimpleDRAM ----
# Expect source at tests/test_simple_dram_read.cpp (create it if missing).
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_simple_dram_read.cpp")
  add_executable(test_simple_dram_read
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_simple_dram_read.cpp"
  )
  target_link_libraries(test_simple_dram_read PRIVATE sfs_core)
  # Header-only nlohmann/json is under include/, no extra link needed.
  sfs_apply_warnings(test_simple_dram_read)
else()
  message(STATUS "tests/test_simple_dram_read.cpp not found; skipping test target.")
endif()

set(SFS_TEST_FIRST_LAYER "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_first_layer_cache.cpp")
if(EXISTS "${SFS_TEST_FIRST_LAYER}")
  add_executable(test_first_layer_cache "${SFS_TEST_FIRST_LAYER}")
  target_link_libraries(test_first_layer_cache PRIVATE sfs_core)
  sfs_apply_warnings(test_first_layer_cache)
else()
  message(STATUS "tests/test_first_layer_cache.cpp not found; skipping first-layer cache sweep test.")
endif()

set(SFS_TEST_LAYER5 "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_layer5_cache.cpp")
if(EXISTS "${SFS_TEST_LAYER5}")
  add_executable(test_layer5_cache "${SFS_TEST_LAYER5}")
  target_link_libraries(test_layer5_cache PRIVATE sfs_core)
  sfs_apply_warnings(test_layer5_cache)
else()
  message(STATUS "tests/test_layer5_cache.cpp not found; skipping layer-5 cache sweep test.")
endif()

set(SFS_TEST_LAYER8 "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_layer8_cache.cpp")
if(EXISTS "${SFS_TEST_LAYER8}")
  add_executable(test_layer8_cache "${SFS_TEST_LAYER8}")
  target_link_libraries(test_layer8_cache PRIVATE sfs_core)
  sfs_apply_warnings(test_layer8_cache)
else()
  message(STATUS "tests/test_layer8_cache.cpp not found; skipping layer-8 cache sweep test.")
endif()

# ---- Optional: Threads example (uncomment if needed) ----
# find_package(Threads REQUIRED)
# target_link_libraries(sfs_core PRIVATE Threads::Threads)

# ---- Optional: Install ----
# install(TARGETS spinalflow-sim RUNTIME DESTINATION bin)
# install(TARGETS sfs_core ARCHIVE DESTINATION lib)

# ---- Notes ----
# - sfs_core contains all project sources except src/main.cpp.
# - spinalflow-sim links against sfs_core to produce the main binary.
# - test_simple_dram_read is a small tool to validate SimpleDRAM reading from a raw image.
# - nlohmann/json.hpp is header-only and expected under include/nlohmann/json.hpp.
