// All comments are in English.

#include "arch/global_merger.hpp"

namespace sf {

bool GlobalMerger::run(Entry& out) {
  // Wiring checks
  if (!fifos_) {
    throw std::runtime_error("GlobalMerger::run: null FIFO array pointer.");
  }

  // Step 1: Check if the Global Merger is allowed to work.
  if (!mfb_.CanGlobalMegerWork()) {
    return false;
  }

  // Step 2: Iterate all IntermediateFIFOs and select the smallest head entry.
  bool found = false;
  std::size_t best_idx = 0;
  Entry best_entry{};

  for (std::size_t i = 0; i < kNumIntermediateFifos; ++i) {
    IntermediateFIFO& fifo = fifos_[i];
    if (fifo.empty()) continue;

    // Peek at the head entry.
    std::optional<Entry> cand = fifo.front();
    if (!cand.has_value()) {
      // Invariant: empty()==false implies front() must have a value.
      throw std::runtime_error("GlobalMerger::run: FIFO not empty but front() returned nullopt.");
    }

    const Entry& e = *cand;
    if (!found) {
      best_entry = e;
      best_idx   = i;
      found      = true;
    } else {
      // Compare by timestamp; tie-break by neuron_id (ascending).
      if (e.ts < best_entry.ts ||
          (e.ts == best_entry.ts && e.neuron_id < best_entry.neuron_id)) {
        best_entry = e;
        best_idx   = i;
      }
    }
  }

  // No available entries in all FIFOs.
  if (!found) {
    return false;
  }

  // Step 3: Pop from the winning FIFO and return the entry.
  IntermediateFIFO& winner = fifos_[best_idx];
  if (!winner.pop()) {
    // Invariant: front() succeeded; pop() must succeed.
    throw std::runtime_error("GlobalMerger::run: FIFO pop() failed unexpectedly.");
  }

  out = best_entry;
  return true;
}

} // namespace sf
