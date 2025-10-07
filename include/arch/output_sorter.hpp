#pragma once
// All comments are in English.

#include <cstddef>
#include <stdexcept>

#include "common/constants.hpp"
#include "common/entry.hpp"
#include "arch/tiled_output_buffer.hpp"
#include "arch/output_spine.hpp"

namespace sf {

/**
 * OutputSorter
 *
 * - Single-entry merge: on each call to Sort(), pick the smallest-ts head
 *   across the 8 tile buffers, push it to OutputSpine, and return true.
 * - If all tile buffers are empty, return false.
 * - Assumes each tile buffer is already monotonically non-decreasing in ts.
 */
class OutputSorter {
public:
  OutputSorter(TiledOutputBuffer* tob, OutputSpine* out_spine)
  : tob_(tob), out_spine_(out_spine) {}

  bool Sort();

private:
  TiledOutputBuffer* tob_ = nullptr; // non-owning
  OutputSpine*       out_spine_ = nullptr; // non-owning
};

} // namespace sf
