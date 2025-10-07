// All comments are in English.

#include "arch/pe_array.hpp"

namespace sf {

bool PEArray::run(FilterBuffer& fb, int h, int w, int W) {
  // Try to fetch one entry from Global Merger.
  if (!gm_.run(gm_entry_)) {
    return false; // no input this cycle
  }

  // We have an input entry. Clear previous outputs for this step.
  out_spike_entries_.clear();

  // Fetch the corresponding weight row (ComputeRowId uses FilterBuffer's members).
  GetWeightRow(fb);

  // Drive all PEs for this step.
  for (std::size_t pe_idx = 0; pe_idx < kNumPE; ++pe_idx) {
    const std::int32_t w_val = static_cast<std::int32_t>(weight_row_[pe_idx]);
    pe_array_[pe_idx].Process(gm_entry_.ts, w_val);

    if (pe_array_[pe_idx].spiked()) {
      Entry e{};
      e.ts = pe_array_[pe_idx].last_ts();
      e.neuron_id = pe_array_[pe_idx].output_neuron_id();
      out_spike_entries_.push_back(e);
    }
  }

  return true; // ran
}

void PEArray::ClearOutputSpikes() {
  out_spike_entries_.clear();
}

} // namespace sf
