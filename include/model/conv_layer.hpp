#pragma once
#include <vector>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <array>

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

/**
 * Build the BatchSpineMap for a single convolution window (one output position).
 *
 * ox, oy: window index in output space (0-based). Window top-left input:
 *         (x0, y0) = (ox*strideW - padW, oy*strideH - padH).
 * kW, kH, inW, inH, strideW, strideH, padW, padH as usual.
 *
 * spineLenEntries: vector of size (inW*inH), giving each logical spine's length (entries).
 *                  Indexing is zero-based: s0 = y*inW + x.
 *
 * Behavior:
 *   - Iterate kH×kW taps; if (x_in,y_in) is out of bounds, skip (padding).
 *   - For in-bounds (x_in,y_in), compute s0 = y_in*inW + x_in.
 *   - Gather ALL block addresses for this s0 via LogicalSpineAllBlockAddrs_0Based(...)
 *     and store them into the BatchSpineMap lane (vector of addresses).
 *   - Split taps into batches of size kNumSpines; last batch may be partial.
 *   - Throws if number of batches exceeds kMaxBatches.
 */
driver::BatchSpineMap BuildBatchMapForWindow(
    int ox, int oy,
    int kW, int kH,
    int inW, int inH,
    int strideW, int strideH,
    int padW, int padH,
    const std::vector<int>& spineLenEntries);

/**
 * OPTIONAL convenience: Build batch maps for all valid windows (row-major by (oy,ox)).
 * If you already loop ox,oy in the caller, you do NOT need this function.
 */
std::vector<driver::BatchSpineMap> BuildAllWindowsBatchMaps(
    int kW, int kH,
    int inW, int inH,
    int strideW, int strideH,
    int padW, int padH,
    const std::vector<int>& spineLenEntries);

/**
 * Pretty-print a BatchSpineMap for verification.
 * Each lane may own multiple block addresses (vector). We show count and a few addresses.
 * Reconstruct zero-based spine id roughly as addr % layerBytes / blockBytes when useful.
 */
std::string PrintBatchMap(const driver::BatchSpineMap& m, int inW);

} // namespace model
} // namespace sf
