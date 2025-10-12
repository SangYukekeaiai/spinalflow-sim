// All comments are in English.

#include "arch/tiled_output_buffer.hpp"
#include "arch/pe_array.hpp"  // requires: out_spike_entries() returning fixed array of optionals
#include <limits>
#include <optional>            // for std::optional

namespace sf {

bool TiledOutputBuffer::run(int tile_id) {
  // Validate tile index.
  if (tile_id < 0 || static_cast<std::size_t>(tile_id) >= kTilesPerSpine) {
    throw std::out_of_range("TiledOutputBuffer::run: tile_id out of range.");
  }

  bool processed = false;
  last_ingested_entries_ = 0;
  last_emitted_entries_ = 0;

  // 1) If any FIFO is full, assert stall for the next cycle; do NOT return here.
  bool any_full = false;
  for (const auto& q : pe_fifos_) {
    if (q.size() >= kLocalFifoDepth) {
      any_full = true;
      break;
    }
  }
  stall_next_cycle_ = any_full;

  // 2) If NOT full, grab the outputs from PEArray and push them to per-PE FIFOs.
  //    Contract: out_spike_entries() -> std::array<std::optional<Entry>, kNumPE>
  if (!any_full) {
    const auto& outs = pe_array_.out_spike_entries();

    bool saw_any = false;
    // One pass: for each PE i, if outs[i] holds a value, push into pe_fifos_[i].
    for (std::size_t i = 0; i < kNumPE; ++i) {
      if (outs[i].has_value()) {
        // Since we checked "any_full" above and each PE contributes at most one
        // entry per cycle, pushing one more cannot overflow beyond depth=4 here.
        pe_fifos_[i].push_back(*outs[i]);
        saw_any = true;
        ++last_ingested_entries_;
      }
    }
    if (saw_any) {
      pe_array_.ClearOutputSpikes();
      processed = true;
    }
  }

  // 3) Pick the smallest-ts among all FIFO heads.
  int best_pe = -1;
  int best_ts = std::numeric_limits<int>::max();

  for (std::size_t i = 0; i < kNumPE; ++i) {
    auto& q = pe_fifos_[i];
    if (!q.empty()) {
      // Replace '.ts' below if your Entry uses a different timestamp field.
      const int ts = static_cast<int>(q.front().ts);
      if (ts < best_ts) {
        best_ts = ts;
        best_pe = static_cast<int>(i);
      }
    }
  }

  // 4) Emit exactly one entry (the smallest ts) to the given tile buffer.
  if (best_pe >= 0) {
    Entry chosen = pe_fifos_[static_cast<std::size_t>(best_pe)].front();
    pe_fifos_[static_cast<std::size_t>(best_pe)].erase(
        pe_fifos_[static_cast<std::size_t>(best_pe)].begin());

    auto& vec = tile_buffers_.at(static_cast<std::size_t>(tile_id));
    if (vec.size() > vec.max_size() - 1) {
      throw std::runtime_error("TiledOutputBuffer::run: tile buffer overflow.");
    }
    vec.push_back(chosen);
    last_emitted_entries_ = 1;
    processed = true;
  }

  return processed;
}

bool TiledOutputBuffer::PeekTileHead(std::size_t tile_id, Entry& out) const {
  if (tile_id >= kTilesPerSpine) return false;
  const auto& vec = tile_buffers_[tile_id];
  if (vec.empty()) return false;
  out = vec.front();
  return true;
}

bool TiledOutputBuffer::PopTileHead(std::size_t tile_id, Entry& out) {
  if (tile_id >= kTilesPerSpine) return false;
  auto& vec = tile_buffers_[tile_id];
  if (vec.empty()) return false;
  out = vec.front();
  vec.erase(vec.begin());
  return true;
}

void TiledOutputBuffer::ClearAll() {
  for (auto& v : tile_buffers_) v.clear();
  for (auto& q : pe_fifos_)    q.clear();
  stall_next_cycle_ = false;
  last_ingested_entries_ = 0;
  last_emitted_entries_ = 0;
}

} // namespace sf
