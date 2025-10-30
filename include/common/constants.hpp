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
inline constexpr int kWeightCacheDefaultSets = 128;
inline constexpr int kWeightCacheDefaultWays = 4;
inline constexpr int kWeightCacheDefaultA1 = 1;
inline constexpr int kWeightCacheDefaultLineBytes = 128;
inline constexpr std::uint64_t kWeightCacheHitLatencyCycles = 1;
inline constexpr std::uint64_t kWeightCacheFillLatencyCycles = 128;

// -----------------------------------------------------------------------------
// Tiled Output Buffer / Output path
// -----------------------------------------------------------------------------
inline constexpr std::size_t kTilesPerSpine          = 8;       // 8 tile buffers per spine
inline constexpr std::size_t kOutputSpineMaxEntries  = 1024;    // double buffer (2 x 512 entries)
inline constexpr std::size_t kMaxSpikesPerStep       = kNumPE;  // worst-case: all PEs spike in a step

// ----------------------------------------------------------------------------- 
// DRAM timing (default assumption; use IOShadow::SetBytesPerCycle to override)
// ----------------------------------------------------------------------------- 
inline constexpr double kDefaultDramBytesPerCycle = 160.0; // 128-bit bus @ 1 cycle per transfer

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
static_assert(kWeightCacheDefaultSets > 0 && (kWeightCacheDefaultSets % 2) == 0,
              "kWeightCacheDefaultSets must be positive and even");
static_assert(kWeightCacheDefaultWays > 0, "kWeightCacheDefaultWays must be positive");
static_assert(kWeightCacheDefaultLineBytes > 0, "kWeightCacheDefaultLineBytes must be positive");

} // namespace sf
