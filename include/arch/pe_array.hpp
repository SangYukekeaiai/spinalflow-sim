#pragma once
// All comments are in English.

#include <array>
#include <cstdint>
#include <vector>

#include "common/constants.hpp"
#include "common/entry.hpp"
#include "arch/global_merger.hpp"   // uses GlobalMerger::run(Entry&)
#include "arch/filter_buffer.hpp"   // FilterBuffer::ComputeRowId/GetRow

namespace sf {

// ===============================
// PE (Processing Element) - local
// ===============================
class PE {
public:
  void RegisterOutputId(std::uint32_t outputId) { output_neuron_id_ = outputId; }
  void SetThreshold(std::int32_t th) { threshold_ = th; }

  void Process(std::uint8_t ts, std::int32_t weight) {
    vmem_ += weight;
    if (vmem_ >= threshold_) {
      vmem_ = 0;
      spiked_ = true;
      last_ts_ = ts;
    } else {
      spiked_ = false;
    }
  }

  bool spiked() const { return spiked_; }
  std::uint32_t output_neuron_id() const { return output_neuron_id_; }
  std::uint8_t last_ts() const { return last_ts_; }

private:
  std::int32_t  vmem_ = 0;
  std::int32_t  threshold_ = 0;
  std::uint32_t output_neuron_id_ = 0;
  bool          spiked_ = false;
  std::uint8_t  last_ts_ = 0;
};


// =======================
// PEArray - top-level PE
// =======================
class PEArray {
public:
  explicit PEArray(GlobalMerger& gm) : gm_(gm) {
    out_spike_entries_.reserve(kMaxSpikesPerStep);
  }

  // Initialize PEs before the outer while-loop of SpinalFlow.
  // output_id = (total_tiles * 128) * (h * W + w) + (tile_idx * 128) + pe_idx
  void InitPEsBeforeLoop(int threshold, int total_tiles, int tile_idx, int h, int w, int W) {
    const int pos_index = h * W + w;
    const std::int64_t stride_pos = static_cast<std::int64_t>(total_tiles) * static_cast<std::int64_t>(kNumPE);
    const std::int64_t base_pos = stride_pos * static_cast<std::int64_t>(pos_index);
    const std::int64_t tile_offset = static_cast<std::int64_t>(tile_idx) * static_cast<std::int64_t>(kNumPE);

    for (std::size_t pe_idx = 0; pe_idx < kNumPE; ++pe_idx) {
      const std::int64_t out_id64 = base_pos + tile_offset + static_cast<std::int64_t>(pe_idx);
      const std::uint32_t out_id  = static_cast<std::uint32_t>(out_id64); // assume fits 32-bit
      pe_array_[pe_idx].RegisterOutputId(out_id);
      pe_array_[pe_idx].SetThreshold(threshold);
    }
    out_spike_entries_.clear();
  }

  // Optional external feeding
  void GetInputEntryFromGM(const Entry& in) { gm_entry_ = in; }

  // Fetch weight row using current gm_entry_.neuron_id and FilterBuffer state.
  void GetWeightRow(FilterBuffer& fb) {
    const int row_id = fb.ComputeRowId(gm_entry_.neuron_id);
    if (row_id >= 0) {
      weight_row_ = fb.GetRow(row_id);
    } else {
      // If padded/invalid tap, zero the row to produce no spikes this step.
      weight_row_.fill(0);
    }
  }

  // Main step: true if the array ran this cycle (GM provided an entry).
  bool run(FilterBuffer& fb, int h, int w, int W);

  // Access the spike outputs produced in the latest run-step.
  const std::vector<Entry>& out_spike_entries() const { return out_spike_entries_; }

  // NEW: Clear the spike outputs after a consumer (e.g., TiledOutputBuffer) copies them.
  void ClearOutputSpikes();

private:
  GlobalMerger& gm_;                                          // reference to GM
  Entry gm_entry_{};                                          // input from GM
  std::array<std::uint8_t, kNumPE> weight_row_{};             // weight row for current computation
  std::array<PE, kNumPE> pe_array_{};                         // 128 PEs
  std::vector<Entry> out_spike_entries_;                      // output spikes for the current step
};

} // namespace sf
