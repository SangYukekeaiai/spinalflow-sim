#include "arch/min_finder_batch.hpp"
#include <limits>

namespace sf {

std::size_t MinFinderBatch::DrainBatchInto(IntermediateFIFO& dst) {
  std::size_t pushed = 0;

  // Keep picking the smallest head among kNumSpines until FIFO is full or all empty.
  while (dst.size() < IntermediateFIFO::kCapacityEntries) {
    int best_lane = -1;
    const Entry* best_head = nullptr;
    std::uint8_t best_ts = std::numeric_limits<std::uint8_t>::max();

    // Linear scan across all physical spines to find the minimum ts head
    for (int lane = 0; lane < kNumSpines; ++lane) {
      const Entry* h = buf_.Head(lane);
      if (!h) continue;

      // Tie-breaker: smaller ts wins; if equal ts, smaller lane index wins.
      if (best_lane < 0 || h->ts < best_ts || (h->ts == best_ts && lane < best_lane)) {
        best_lane  = lane;
        best_ts    = h->ts;
        best_head  = h;   // cache the pointer to avoid calling Head() twice
      }
    }

    // No available heads across all spines.
    if (best_lane < 0) break;

    // Push the selected entry and pop it from its lane.
    const Entry e = *best_head;
    if (!buf_.PopHead(best_lane)) {
      // This should not happen because we just saw a valid head.
      break;
    }

    if (!dst.push(e)) {
      // FIFO is full; stop draining.
      break;
    }
    ++pushed;
  }

  return pushed;
}

bool MinFinderBatch::DrainOneInto(IntermediateFIFO& dst) {
  if (dst.full()) return false;

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

  if (best_lane < 0 || !best_head) return false;
  if (!buf_.PopHead(best_lane)) return false;
  if (!dst.push(*best_head)) return false;
  return true;
}

} // namespace sf
