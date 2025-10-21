// All comments are in English.

#include "arch/pe_array.hpp"

namespace sf {

bool PEArray::run(FilterBuffer& fb) {
  last_cache_result_ = {};
  // Try to fetch one entry from Global Merger.
  if (!gm_.run(gm_entry_)) {
    last_row_lookup_.reset();
    return false;
  }

  // We have an input entry. Reset output slots for this step.
  ResetOutputSlots();

  // Fetch the corresponding weight row (ComputeRowId uses FilterBuffer's members).
  GetWeightRow(fb);

  if (cache_) {
    if (last_row_lookup_.has_value() && current_tile_idx_ >= 0) {
      const auto& info = last_row_lookup_.value();
      // Inform cache of the current time step before scoring/eviction.
      cache_->BeginTimeStep(static_cast<int>(gm_entry_.ts));
      cache_->NotifySpike(info.c_in);
      const sf::arch::cache::LineAddr addr(
          static_cast<std::uint32_t>(current_tile_idx_),
          static_cast<std::uint32_t>(info.c_in),
          static_cast<std::uint32_t>(info.kh),
          static_cast<std::uint32_t>(info.kw));
      last_cache_result_ = cache_->Access(addr);
    }
  }

  // Drive all PEs for this step.
  for (std::size_t pe_idx = 0; pe_idx < kNumPE; ++pe_idx) {
    // Decode weight to float (fixed-point or scale)
    float w_float = DecodeWeightToFloat(weight_row_[pe_idx]);
    pe_array_[pe_idx].Process(gm_entry_.ts, w_float);
    if (pe_array_[pe_idx].spiked()) {
      Entry e{};
      e.ts        = pe_array_[pe_idx].last_ts();
      e.neuron_id = pe_array_[pe_idx].output_neuron_id();
      out_spike_entries_[pe_idx] = e;          // set this PE's slot
    } else {
      out_spike_entries_[pe_idx] = std::nullopt; // explicitly empty
    }
  }

  return true; // ran
}

void PEArray::ClearOutputSpikes() {
  // Clear all per-PE slots to empty.
  ResetOutputSlots();
}

} // namespace sf
