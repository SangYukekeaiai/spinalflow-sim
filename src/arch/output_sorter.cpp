// All comments are in English.

#include "arch/output_sorter.hpp"

namespace sf {

bool OutputSorter::Sort() {
  if (!tob_ || !out_spine_) {
    throw std::runtime_error("OutputSorter::Sort: null dependency.");
  }

  // Scan heads of the 8 tile buffers and pick the smallest timestamp.
  bool found = false;
  std::size_t best_idx = 0;
  Entry best{};

  for (std::size_t i = 0; i < kTilesPerSpine; ++i) {
    Entry head{};
    if (!tob_->PeekTileHead(i, head)) continue; // empty tile buffer

    if (!found || head.ts < best.ts) {
      best = head;
      best_idx = i;
      found = true;
    }
  }

  if (!found) {
    return false; // all empty
  }

  // Pop from the winning tile buffer and push to OutputSpine.
  Entry popped{};
  if (!tob_->PopTileHead(best_idx, popped)) {
    // Invariant: peek succeeded; pop must succeed.
    throw std::runtime_error("OutputSorter::Sort: pop failed unexpectedly.");
  }

  // Push (may throw on capacity exceeded).
  out_spine_->Push(popped);
  return true;
}

} // namespace sf
