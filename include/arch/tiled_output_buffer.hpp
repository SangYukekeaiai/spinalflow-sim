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
 * - Adds per-PE local FIFOs (depth=4) to temporarily hold PE outputs.
 * - Input contract (new): PEArray::out_spike_entries() returns a fixed-size
 *   std::array<std::optional<Entry>, kNumPE>, one "slot" per PE per cycle.
 * - "Tile" here means an output-spine partition (e.g., every 128 output channels).
 * - The caller passes tile_id on each run(...) to choose which tile buffer to append to.
 * - Stall policy (step 1): if any local FIFO is full, set stall_next_cycle_=true.
 *   We still emit one entry from the existing FIFO heads in the same run to avoid stalling the pipeline.
 */
namespace sf {

class TiledOutputBuffer {
public:
  explicit TiledOutputBuffer(PEArray& pe_array)
  : pe_array_(pe_array) {}

  // Returns true if anything happened (ingested from PEArray and/or emitted to a tile,
  // or stall flag updated).
  bool run(int tile_id);

  bool PeekTileHead(std::size_t tile_id, Entry& out) const;
  bool PopTileHead (std::size_t tile_id, Entry& out);

  void ClearAll();

  bool stall_next_cycle() const { return stall_next_cycle_; }

private:
  static constexpr std::size_t kLocalFifoDepth = 4;

  PEArray& pe_array_;
  bool stall_next_cycle_ = false;

  // Per-PE local FIFOs (front at index 0).
  std::array<std::vector<Entry>, kNumPE> pe_fifos_{};

  // kTilesPerSpine per-tile buffers (front at index 0).
  std::array<std::vector<Entry>, kTilesPerSpine> tile_buffers_;
};

} // namespace sf
