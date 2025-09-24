#include "arch/global_merger.hpp"
#include <limits>

namespace sf {

std::optional<GlobalMerger::PickResult>
GlobalMerger::PickAndPop(const std::array<IntermediateFIFO*, kMaxBatches>& fifos) {
  int best_idx = -1;
  const Entry* best_entry = nullptr;
  std::uint8_t best_ts = std::numeric_limits<std::uint8_t>::max();

  // Scan all candidate FIFOs
  for (int i = 0; i < kMaxBatches; ++i) {
    if (!fifos[i]) continue;  // null pointer check
    auto h = fifos[i]->front();
    if (!h) continue;         // empty fifo
    const std::uint8_t ts = h->ts;

    if (best_idx < 0 || ts < best_ts || (ts == best_ts && i < best_idx)) {
      best_idx   = i;
      best_ts    = ts;
      best_entry = &(*h); // cache pointer to avoid second .front() later
    }
  }

  if (best_idx < 0 || !best_entry) return std::nullopt;

  // Pop from the chosen FIFO and return the cached entry.
  auto* fifo = fifos[best_idx];
  Entry e = *best_entry;
  if (!fifo || !fifo->pop()) return std::nullopt;
  return PickResult{e, best_idx};
}

} // namespace sf
