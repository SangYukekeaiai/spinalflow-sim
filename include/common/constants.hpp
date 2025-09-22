#pragma once
#include <cstddef>
#include <cstdint>

namespace sf {

/**
 * Common system-wide constants used across the SpinalFlow simulator.
 * Keep all global geometry/sizing parameters here so that every
 * module (arch, driver, core, etc.) stays consistent.
 */

// Number of physical input spines available in the architecture
constexpr int kNumSpines = 16;

// Capacity of each physical input spine (entries per spine FIFO)
constexpr int kCapacityPerSpine = 128;

// Maximum number of batches supported in CPU-side scheduling
constexpr int kMaxBatches = 4;

} // namespace sf
