#pragma once
#include <array>
#include <optional>
#include "common/entry.hpp"
#include "arch/intermediate_fifo.hpp"

namespace sf {

/**
 * GlobalMerger
 *
 * Performs a four-way merge across IntermediateFIFO instances. On each call,
 * it selects the globally smallest ts among the four FIFOs (tie-breaking by
 * lower FIFO index), pops it from that FIFO, and returns the entry.
 */
class GlobalMerger {
public:
  struct PickResult {
    Entry entry;
    int   fifo_index;
  };

  // Pick and pop the globally smallest entry across the provided FIFOs.
  // Return std::nullopt if all FIFOs are empty.
  static std::optional<PickResult> PickAndPop(const std::array<IntermediateFIFO*, 4>& fifos);
};

} // namespace sf
