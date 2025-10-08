// common/constants.hpp
#pragma once
// All comments are in English.

#include <cstddef>
#include <cstdint>

namespace sf {

// -----------------------------------------------------------------------------
// ISB (InputSpineBuffer)
// -----------------------------------------------------------------------------
inline constexpr int kNumPhysISB = 16;
inline constexpr int kIsbEntries = 2048;

// -----------------------------------------------------------------------------
// Intermediate FIFO
// -----------------------------------------------------------------------------
inline constexpr std::size_t kInterFifoCapacityBytes = 256;
inline constexpr std::size_t kNumIntermediateFifos   = 4;

// -----------------------------------------------------------------------------
// PEs / Filter Buffer
// -----------------------------------------------------------------------------
inline constexpr std::size_t kNumPE      = 128;   // weights per row / PEs per array
inline constexpr std::size_t kFilterRows = 4608;  // total rows stored in FilterBuffer

// -----------------------------------------------------------------------------
// Tiled Output Buffer / Output path
// -----------------------------------------------------------------------------
inline constexpr std::size_t kTilesPerSpine          = 8;       // 8 tile buffers per spine
inline constexpr std::size_t kOutputSpineMaxEntries  = 65536;   // capacity limit for OutputSpine buffer
inline constexpr std::size_t kMaxSpikesPerStep       = kNumPE;  // worst-case: all PEs spike in a step

// -----------------------------------------------------------------------------
// Sanity checks
// -----------------------------------------------------------------------------
static_assert(kNumPhysISB  > 0,  "kNumPhysISB must be positive");
static_assert(kIsbEntries  > 0,  "kIsbEntries must be positive");
static_assert(kInterFifoCapacityBytes > 0, "kInterFifoCapacityBytes must be positive");
static_assert(kNumIntermediateFifos > 0,   "kNumIntermediateFifos must be positive");
static_assert(kNumPE > 0,                  "kNumPE must be positive");
static_assert(kFilterRows > 0,             "kFilterRows must be positive");
static_assert(kTilesPerSpine > 0,          "kTilesPerSpine must be positive");
static_assert(kOutputSpineMaxEntries > 0,  "kOutputSpineMaxEntries must be positive");

} // namespace sf
