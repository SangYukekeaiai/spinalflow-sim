#pragma once
#include <array>
#include <optional>
#include "common/constants.hpp"
#include "common/entry.hpp"
#include "arch/intermediate_fifo.hpp"

namespace sf {

/**
 * GlobalMerger
 *
 * Performs an up-to-kMaxBatches-way merge across IntermediateFIFO instances.
 * On each call, it selects the globally smallest ts (tie-breaking by lower
 * FIFO index), pops it from that FIFO, and returns the entry.
 */
class GlobalMerger {
public:
  struct PickResult {
    Entry entry;
    int   fifo_index;
  };

  // Pick and pop the globally smallest entry across the provided FIFOs.
  // Return std::nullopt if all FIFOs are empty.
  static std::optional<PickResult> PickAndPop(const std::array<IntermediateFIFO*, kMaxBatches>& fifos);
};

} // namespace sf
