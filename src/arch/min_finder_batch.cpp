#include "arch/min_finder_batch.hpp"
#include <limits>
#include "core/clock.hpp" // to access core FIFOs and control state

namespace sf {

std::optional<Entry> MinFinderBatch::PopMinHead() {
  int best_lane = -1;
  const Entry* best_head = nullptr;
  std::uint8_t best_ts = std::numeric_limits<std::uint8_t>::max();

  for (int lane = 0; lane < kNumSpines; ++lane) {
    const Entry* h = buf_.Head(lane);
    if (!h) continue;
    if (best_lane < 0 || h->ts < best_ts || (h->ts == best_ts && lane < best_lane)) {
      best_lane = lane;
      best_ts = h->ts;
      best_head = h;
    }
  }

  if (best_lane < 0 || !best_head) return std::nullopt;

  const Entry result = *best_head;
  if (!buf_.PopHead(best_lane)) return std::nullopt;
  return result;
}

std::size_t MinFinderBatch::DrainBatchInto(IntermediateFIFO& dst) {
  std::size_t pushed = 0;
  while (!dst.full()) {
    auto next = PopMinHead();
    if (!next.has_value()) break;
    if (!dst.push(*next)) break;
    ++pushed;
  }
  return pushed;
}

bool MinFinderBatch::DrainOneInto(IntermediateFIFO& dst) {
  if (dst.full()) return false;
  auto next = PopMinHead();
  if (!next.has_value()) return false;
  return dst.push(*next);
}

// ---- Stage-4 integration ----
bool MinFinderBatch::run() {
  if (!core_) return false;

  // 1) Upstream (S3->S4) invalid => propagate invalid to S5 and stall.
  if (!core_->st3_st4_valid()) {
    if (core_->st4_st5_valid()) core_->SetSt4St5Valid(false);
    return false;
  }
  else {
    if (!core_->st4_st5_valid()) core_->SetSt4St5Valid(true);
  }

  const int batches = core_->batches_needed();
  int cursor = core_->load_batch_cursor();
  if (cursor < 0 || cursor >= batches) return false;

  auto& fifo = core_->fifos()[cursor];
  if (fifo.full()) return false;

  // 1) First attempt: try to move one entry from ACTIVE heads.
  if (DrainOneInto(fifo)) {
    return true;
  }

  // 2) No move: proactively swap SHADOW->ACTIVE for any spine whose ACTIVE is empty.
  //    This avoids stalling one extra cycle waiting for Stage-5 to do the swap.
  bool swapped_any = false;
  for (int s = 0; s < kNumSpines; ++s) {
    // If ACTIVE is empty, try to bring SHADOW up.
    if (buf_.Empty(s)) {
      // SwapToShadow() returns true only if shadow had data.
      if (buf_.SwapToShadow(s)) {
        swapped_any = true;
      }
    }
  }

  // 3) If we swapped anything, try one more time to move an entry.
  if (swapped_any) {
    if (DrainOneInto(fifo)) {
      return true;
    }
  }

  // 4) Still nothing moved: decide whether the current batch's inputs are truly drained.
  //    Criteria for "truly drained" on every spine:
  //      - ACTIVE is empty (already true by construction here),
  //      - SHADOW is also empty (if SHADOW had data, step 2 would have swapped it),
  //      - AND meta.fully_loaded == 1 (no more segments remain in DRAM for this spine).
  //
  //    We don't have a direct "shadow empty" query; step 2's SwapToShadow() attempt
  //    ensures that if SHADOW had data, it would now be in ACTIVE (and DrainOneInto above
  //    would have succeeded or will succeed next time). So at this point, ACTIVE empty
  //    implies SHADOW is empty too (or we already tried swapping but it was empty).
  bool all_spines_fully_drained = true;

  for (int s = 0; s < kNumSpines; ++s) {
    // If ACTIVE is non-empty for any spine, we are definitely not drained.
    if (!buf_.Empty(s)) { all_spines_fully_drained = false; break; }

    // If DRAM hasn't fully loaded all segments for this spine, not drained yet.
    const auto& meta = buf_.LaneMeta(s);
    if (!meta.fully_loaded) {
      all_spines_fully_drained = false;
      break;
    }

    // Optionally tighten: mark fully_drained when applicable (no effect on decision).
    // buf_.MarkFullyDrainedIfApplicable(s);
  }

  if (all_spines_fully_drained) {
    // Now it's safe to declare the INPUT for this batch fully drained.
    core_->SetInputDrained(cursor, true);
    core_->AdvanceLoadBatchCursor();
  }

  return false;
}


} // namespace sf
