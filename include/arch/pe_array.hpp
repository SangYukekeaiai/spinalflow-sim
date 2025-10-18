#pragma once
// All comments are in English.

#include <array>
#include <cstdint>
#include <vector>
#include <optional>               // NEW

#include "common/constants.hpp"
#include "common/entry.hpp"
#include <cmath>                    // std::ldexp
#include "arch/global_merger.hpp"   // uses GlobalMerger::run(Entry&)
#include "arch/filter_buffer.hpp"   // FilterBuffer::ComputeRowId/GetRow
#include "arch/cache/cache.hpp"
#include <iostream>

namespace sf {

// ===============================
// PE (Processing Element) - local
// ===============================
class PE {
public:
  void RegisterOutputId(std::uint32_t outputId) { output_neuron_id_ = outputId; }
  void SetThreshold(float th) { threshold_ = th; }

  void Process(std::int8_t ts, float weight) {
    vmem_ += weight;
    if (vmem_ >= threshold_) {
      vmem_ = 0.0f;
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
  float  vmem_ = 0.0f;
  float  threshold_ = 1.0f;
  std::uint32_t output_neuron_id_ = 0;
  bool          spiked_ = false;
  std::uint8_t  last_ts_ = 0;
};


// =======================
// PEArray - top-level PE
// =======================
class PEArray {
public:
  explicit PEArray(GlobalMerger& gm, sf::arch::cache::CacheSim* cache = nullptr)
      : gm_(gm), cache_(cache) {
    // No reserve needed; we use a fixed array of optionals.
    ResetOutputSlots();
  }

  void AttachCache(sf::arch::cache::CacheSim* cache) { cache_ = cache; }
  int current_tile_idx() const { return current_tile_idx_; }
  int last_cache_cycles() const { return last_cache_result_.demand_cycles; }
  bool last_cache_miss() const { return last_cache_result_.demand_miss; }
  const sf::arch::cache::AccessResult& last_cache_result() const { return last_cache_result_; }

  void SetWeightParamsAndThres(float threshold, int w_bits, bool w_signed, int w_frac_bits, float w_scale) {
    for (std::size_t pe_idx = 0; pe_idx < kNumPE; ++pe_idx) {
      pe_array_[pe_idx].SetThreshold(threshold);
    }
    w_bits_ = w_bits;
    w_signed_ = w_signed;
    w_frac_bits_ = w_frac_bits;
    w_scale_ = w_scale;
  }

  // Initialize PEs before the outer while-loop of SpinalFlow.
  // output_id = (total_tiles * 128) * (h * W + w) + (tile_idx * 128) + pe_idx
  void InitPEsOutputNIDBeforeLoop(int total_tiles, int tile_idx, int h, int w, int W) {
    const int pos_index = h * W + w;
    const std::int64_t stride_pos = static_cast<std::int64_t>(total_tiles) * static_cast<std::int64_t>(kNumPE);
    const std::int64_t base_pos = stride_pos * static_cast<std::int64_t>(pos_index);
    const std::int64_t tile_offset = static_cast<std::int64_t>(tile_idx) * static_cast<std::int64_t>(kNumPE);

    for (std::size_t pe_idx = 0; pe_idx < kNumPE; ++pe_idx) {
      const std::int64_t out_id64 = base_pos + tile_offset + static_cast<std::int64_t>(pe_idx);
      const std::uint32_t out_id  = static_cast<std::uint32_t>(out_id64); // assume fits 32-bit
      pe_array_[pe_idx].RegisterOutputId(out_id);
    }
    ResetOutputSlots(); // was: out_spike_entries_.clear();
    current_tile_idx_ = tile_idx;
  }

  inline float DecodeWeightToFloat(std::int8_t wq) const noexcept {
    if (w_frac_bits_ >= 0) {
      // std::ldexp(1.0f, -n) == 2^-n
      return static_cast<float>(wq) * std::ldexp(1.0f, -w_frac_bits_);
    }
    // Fallback to provided scale (should be 2^-n in your exporter)
    return static_cast<float>(wq) * ((w_scale_ > 0.0f) ? w_scale_ : 1.0f);
  }

  // Optional external feeding
  void GetInputEntryFromGM(const Entry& in) { gm_entry_ = in; }

  // Fetch weight row using current gm_entry_.neuron_id and FilterBuffer state.
  void GetWeightRow(FilterBuffer& fb) {
    last_row_lookup_ = fb.ResolveRow(gm_entry_.neuron_id);
    if (last_row_lookup_.has_value()) {
      weight_row_ = fb.GetRow(last_row_lookup_->row_id);
    } else {
      // If padded/invalid tap, zero the row to produce no spikes this step.
      std::cout << "PEArray::GetWeightRow: Padding/invalid tap for neuron_id " << gm_entry_.neuron_id << ", zeroing weight row.\n";
      weight_row_.fill(0);
    }
  }

  // Main step: true if the array ran this cycle (GM provided an entry).
  bool run(FilterBuffer& fb);

  // Access the spike outputs produced in the latest run-step.
  // NEW: fixed array with one optional Entry per PE.
  const std::array<std::optional<Entry>, kNumPE>& out_spike_entries() const { return out_spike_entries_; }

  // Clear the spike outputs after a consumer copies them.
  void ClearOutputSpikes();

private:
  // Helper to reset all output slots to empty.
  void ResetOutputSlots() {
    for (auto& s : out_spike_entries_) s.reset();
  }

  GlobalMerger& gm_;                                          // reference to GM
  Entry gm_entry_{};                                          // input from GM
  std::array<std::int8_t, kNumPE> weight_row_{};              // weight row for current computation
  std::array<PE, kNumPE> pe_array_{};                         // 128 PEs

  // NEW: one optional Entry per PE for the current step.
  std::array<std::optional<Entry>, kNumPE> out_spike_entries_{};

  sf::arch::cache::CacheSim* cache_ = nullptr;               // shared cache simulator
  int current_tile_idx_ = -1;
  std::optional<FilterBuffer::RowLookup> last_row_lookup_;
  sf::arch::cache::AccessResult last_cache_result_{};

  int w_bits_ = 8;                                           // weight bit-width
  bool w_signed_ = true;                                     // weight signedness
  int w_frac_bits_ = 0;                                      // weight fractional bits (for fixed-point)
  float w_scale_ = 1.0f;                                     // weight scale (real multiplier)
};

} // namespace sf
