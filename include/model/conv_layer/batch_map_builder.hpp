#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include "common/constants.hpp"
#include "common/entry.hpp"
#include "arch/driver/batch_spine_map.hpp"

namespace sf {
namespace model {

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
 *   - Gather ALL block addresses for this s0 and store into per-lane vectors.
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
 */
std::vector<driver::BatchSpineMap> BuildAllWindowsBatchMaps(
    int kW, int kH,
    int inW, int inH,
    int strideW, int strideH,
    int padW, int padH,
    const std::vector<int>& spineLenEntries);

} // namespace model
} // namespace sf
