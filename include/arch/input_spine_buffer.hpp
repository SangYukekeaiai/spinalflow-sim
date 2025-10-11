#pragma once
// All comments are in English.

#include <cstdint>
#include <vector>
#include <stdexcept>
#include <limits>
#include <algorithm>

#include "common/constants.hpp"  // expects kNumPhysISB, kIsbEntries
#include "common/entry.hpp"      // sf::Entry

namespace sf { namespace dram {
  // Forward-declare SimpleDRAM to avoid forcing include path here.
  class SimpleDRAM;
}} // namespace sf::dram

namespace sf {

/**
 * InputSpineBuffer
 *
 * A fixed number of physical input-spine buffers (no shadow/active).
 * Each physical buffer holds an array of `Entry` (timestamp + neuron_id).
 * Data are block-loaded from DRAM by logical spine id, one batch at a time.
 *
 * Public API matches the specification you gave:
 *  1) PreloadFirstBatch(logical_spine_ids_first_batch, layer_id)
 *  2) run(logical_spine_ids_current_batch, layer_id, current_batch_cursor, total_batches_needed)
 *  3) PopSmallestTsEntry(out)
 */
class InputSpineBuffer {
public:
  // Construct with a DRAM handle; sizes come from common/constants.hpp.
  explicit InputSpineBuffer(sf::dram::SimpleDRAM* dram);

  struct Timing {
    uint32_t bw_bytes_per_cycle = 16; // e.g., 128b/cycle by default
    uint32_t fixed_latency = 0;       // per-load fixed cycles (DMA setup)
    uint32_t wire_entry_bytes = 5;    // ts:uint8 + nid:uint32 on the wire
    uint32_t parallel_loads = 1;      // number of loads that can progress in parallel
  };

  void SetTiming(const Timing& t) { timing_ = t; }

  // Reset all buffers to empty (helper; not required by your spec but useful).
  void Reset();

  // (A) Pre-load the first batch into the physical buffers.
  // Returns true if load happened, false if the input list is empty.
  bool PreloadFirstBatch(const std::vector<int>& logical_spine_ids_first_batch,
                         int layer_id, uint64_t* out_cycles = nullptr);

  // (B) Run-time loader: if all buffers are empty and batches remain,
  // load the current batch into physical buffers.
  // Returns true if a load happened this call; false otherwise.
  bool run(const std::vector<int>& logical_spine_ids_current_batch,
           int layer_id,
           int current_batch_cursor,
           int total_batches_needed,
           uint64_t* out_cycles);

  // (C) Pop the Entry with the globally-smallest timestamp among all buffers.
  // Returns true if an entry was popped; false if all buffers are empty.
  bool PopSmallestTsEntry(Entry& out);

  // Utility: whether all physical buffers are empty.
  bool AllEmpty() const;

  // Expose configured sizes (from constants).
  int NumPhysBuffers() const { return num_phys_; }
  int EntriesPerBuffer() const { return entries_per_buf_; }

private:
  // Load a batch of logical spine ids into the physical buffers.
  // The list size must be <= number of physical buffers.
  uint64_t LoadBatchIntoBuffers_(const std::vector<int>& logical_spine_ids,
                             int layer_id);

  static inline uint64_t CeilDivU64(uint64_t a, uint64_t b) {
    return (a + b - 1) / b;
  }

  // Compute available entries in buffer i.
  int Available_(int i) const {
    return valid_count_[static_cast<size_t>(i)] - read_idx_[static_cast<size_t>(i)];
  }

private:
  // Hardware-configured sizes (taken from common/constants.hpp).
  const int num_phys_;
  const int entries_per_buf_;
  const std::size_t bytes_per_buf_;

  // Flat storage for physical buffers: buffers_[i][j] is the j-th entry in i-th buffer.
  std::vector<std::vector<Entry>> buffers_;

  // Per-buffer read pointer (in entries).
  std::vector<int> read_idx_;

  // Per-buffer valid entries count (0..entries_per_buf_).
  std::vector<int> valid_count_;

  // Track which logical spine id currently resides in each physical buffer (-1 if empty).
  std::vector<int> logical_id_loaded_;

  // DRAM interface for table-driven memcpy loads.
  sf::dram::SimpleDRAM* dram_ = nullptr;

  Timing timing_;
};

} // namespace sf
