#include "arch/min_finder_batch.hpp"
#include <limits>

namespace sf {

std::optional<Entry> MinFinderBatch::PopMinHead() {
  int best_lane = -1;
  const Entry* best_head = nullptr;
  std::uint8_t best_ts = std::numeric_limits<std::uint8_t>::max();

  // Linear scan across all physical spines to find the minimum ts head
  for (int lane = 0; lane < kNumSpines; ++lane) {
    const Entry* h = buf_.Head(lane);
    if (!h) continue;

    // Tie-breaker: smaller ts wins; if equal ts, smaller lane index wins.
    if (best_lane < 0 || h->ts < best_ts || (h->ts == best_ts && lane < best_lane)) {
      best_lane = lane;
      best_ts = h->ts;
      best_head = h; // cache the pointer to avoid calling Head() twice
    }
  }

  if (best_lane < 0 || !best_head) return std::nullopt;

  const Entry result = *best_head;
  if (!buf_.PopHead(best_lane)) return std::nullopt;
  return result;
}

std::size_t MinFinderBatch::DrainBatchInto(IntermediateFIFO& dst) {
  std::size_t pushed = 0;

  // Keep picking the smallest head among kNumSpines until FIFO is full or all empty.
  while (!dst.full()) {
    auto next = PopMinHead();
    if (!next.has_value()) break;
    if (!dst.push(*next)) break; // should not happen if capacity check is accurate
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

} // namespace sf
