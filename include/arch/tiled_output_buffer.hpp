#pragma once
// All comments are in English.

#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "common/constants.hpp"
#include "common/entry.hpp"

namespace sf {
class PEArray; // forward declaration; actual header is included in the .cpp
}

/**
 * TiledOutputBuffer
 *
 * - Aggregates output spikes from a PEArray into per-tile buffers.
 * - "Tile" here means an output-spine partition (e.g., every 128 output channels).
 * - This class does NOT store a tile_id member. The caller passes tile_id
 *   on each run(...) to choose which tile buffer to append to.
 * - A simple bandwidth/latency model is implemented via writeback_cooldown_cycles_:
 *   * When > 0, run(...) only decrements the cooldown (returns true).
 *   * When == 0 and PEArray has outputs, copy them into tile_buffers_[tile_id],
 *     set cooldown = number of copied entries, clear PEArray outputs, and return true.
 *   * Otherwise, return false.
 */
namespace sf {

class TiledOutputBuffer {
public:
  explicit TiledOutputBuffer(PEArray& pe_array)
  : pe_array_(pe_array) {}

  // Main step: copy PE outputs into tile_buffers_[tile_id] with a cooldown model.
  // Returns true if anything happened (cooldown decremented or outputs copied).
  bool run(int tile_id);

  // Expose heads to OutputSorter (global merge assumes per-tile non-decreasing ts).
  bool PeekTileHead(std::size_t tile_id, Entry& out) const;
  bool PopTileHead (std::size_t tile_id, Entry& out);

  // Optional helper to hard-reset all internal tile buffers.
  void ClearAll();

  // Accessor for the current cooldown value.
  int writeback_cooldown_cycles() const { return writeback_cooldown_cycles_; }

private:
  PEArray& pe_array_;   // source of out_spike_entries + ClearOutputSpikes()
  int writeback_cooldown_cycles_ = 0;

  // kTilesPerSpine per-tile buffers (front at index 0).
  // NOTE: "Tile" = output-spine partition, not the number of PEs.
  std::array<std::vector<Entry>, kTilesPerSpine> tile_buffers_;
};

} // namespace sf
