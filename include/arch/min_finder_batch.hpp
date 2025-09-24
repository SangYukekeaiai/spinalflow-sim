#pragma once
#include <optional>
#include "common/constants.hpp"
#include "arch/input_spine_buffer.hpp"
#include "arch/intermediate_fifo.hpp"

namespace sf {

/**
 * MinFinderBatch
 *
 * Selects the minimum timestamp head across all physical spines and pushes
 * entries into an IntermediateFIFO in non-decreasing ts order. It consumes
 * ONLY the ACTIVE bank of each spine. The caller is responsible for calling
 * SwapToShadow() on the InputSpineBuffer when appropriate.
 */
class MinFinderBatch {
public:
  explicit MinFinderBatch(InputSpineBuffer& buf) : buf_(buf) {}

  // Drain until the FIFO is full or all ACTIVE heads are empty.
  // Returns the number of entries pushed to 'dst'.
  std::size_t DrainBatchInto(IntermediateFIFO& dst);

  // Drain at most one entry (if available) into dst. Returns true on success.
  bool DrainOneInto(IntermediateFIFO& dst);

private:
  InputSpineBuffer& buf_;
  std::optional<Entry> PopMinHead();
};

} // namespace sf
