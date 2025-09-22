#include "arch/output_harvest.hpp"
#include <stdexcept>

namespace sf {

OutputHarvest::OutputHarvest()
  : index_fifo_(kNumPEs) {
    Clear();
}

void OutputHarvest::Clear() {
    for (int i = 0; i < kNumPEs; ++i) {
        produced_[i] = false;
    }
    count_ = 0;
    index_fifo_.Clear();
}

bool OutputHarvest::Capture(int pe_id, const OutputToken& tok) {
    if (pe_id < 0 || pe_id >= kNumPEs) return false;
    if (produced_[pe_id]) return false; // already captured once this step
    slots_[pe_id]   = tok;
    produced_[pe_id]= true;
    ++count_;
    // Push PE index to the IndexFIFO for O(outputs) traversal later.
    (void)index_fifo_.Push(pe_id);
    return true;
}

bool OutputHarvest::Has(int pe_id) const {
    if (pe_id < 0 || pe_id >= kNumPEs) return false;
    return produced_[pe_id];
}

const OutputToken& OutputHarvest::Get(int pe_id) const {
    if (pe_id < 0 || pe_id >= kNumPEs) {
        throw std::out_of_range("OutputHarvest::Get: pe_id out of range");
    }
    if (!produced_[pe_id]) {
        throw std::logic_error("OutputHarvest::Get: token not present for this PE");
    }
    return slots_[pe_id];
}

} // namespace sf
