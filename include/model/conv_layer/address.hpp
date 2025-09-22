#pragma once
#include <vector>
#include <cstddef>
#include <cstdint>
#include "common/constants.hpp"
#include "common/entry.hpp"
#include "arch/driver/batch_spine_map.hpp"

namespace sf {
namespace model {

/** Bytes per DRAM block for one logical spine chunk (128 entries * sizeof(Entry)). */
inline constexpr std::size_t SpineBlockBytes() {
  return static_cast<std::size_t>(kCapacityPerSpine) * sizeof(Entry);
}

/**
 * Compute all DRAM base addresses (layered, interleaved across spines) for a
 * zero-based logical spine id s0 in [0 .. totalSpines-1], given its length (entries).
 *
 * Layout:
 *   For block i (0-based), the address is:
 *     addr_i = i * (totalSpines * blockBytes) + s0 * blockBytes
 * where blockBytes = kCapacityPerSpine * sizeof(Entry),
 * and numBlocks = ceil(lenEntries / kCapacityPerSpine).
 */
std::vector<driver::BatchSpineMap::Addr>
LogicalSpineAllBlockAddrs_0Based(int s0, int totalSpines, int lenEntries);

} // namespace model
} // namespace sf
