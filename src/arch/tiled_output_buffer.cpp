// All comments are in English.

#include "arch/tiled_output_buffer.hpp"
#include "arch/pe_array.hpp"  // requires: out_spike_entries() and ClearOutputSpikes()

namespace sf {

bool TiledOutputBuffer::run(int tile_id) {
  // Validate tile index.
  if (tile_id < 0 || static_cast<std::size_t>(tile_id) >= kTilesPerSpine) {
    throw std::out_of_range("TiledOutputBuffer::run: tile_id out of range.");
  }

  // If in cooldown, decrement and report activity.
  if (writeback_cooldown_cycles_ > 0) {
    --writeback_cooldown_cycles_;
    return true;
  }

  // Otherwise, check if PEArray produced outputs for this step.
  const auto& outs = pe_array_.out_spike_entries();
  if (outs.empty()) {
    return false; // nothing to do
  }

  // Insert all entries into the tile buffer for the given tile_id.
  auto& vec = tile_buffers_.at(static_cast<std::size_t>(tile_id));

  // Optional capacity check to detect pathological overflow.
  if (vec.size() > vec.max_size() - outs.size()) {
    throw std::runtime_error("TiledOutputBuffer::run: tile buffer overflow.");
  }

  const std::size_t prev_sz = vec.size();
  vec.insert(vec.end(), outs.begin(), outs.end());
  const std::size_t appended = vec.size() - prev_sz;

  // Set cooldown equal to the number of entries copied (simple latency model).
  writeback_cooldown_cycles_ = static_cast<int>(appended);

  // Clear the PEArray outputs after successful copy.
  pe_array_.ClearOutputSpikes();

  return true;
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
  // vector pop-front: erase(begin) is O(n), but kTilesPerSpine is small and
  // the sorter pops one at a time, which is acceptable per the spec.
  vec.erase(vec.begin());
  return true;
}

void TiledOutputBuffer::ClearAll() {
  for (auto& v : tile_buffers_) v.clear();
  writeback_cooldown_cycles_ = 0;
}

} // namespace sf
