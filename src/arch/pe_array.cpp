#include "arch/pe_array.hpp"

// All comments are in English.
#include <limits>
#include "core/clock.hpp"              // ClockCore API (st1_st2_valid, ts_picker)
#include "arch/smallest_ts_picker.hpp" // SmallestTsPicker::Stage2Write()

namespace sf {

// ===== PE implementation =====

PE::PE()
    : V_mem_(0), spiked_(false), reset_V_mem_(0), out_neuron_id_(0xFFFFFFFFu) {}

int8_t PE::Accumulator(int8_t V_mem, int8_t filter) const {
    int16_t sum = static_cast<int16_t>(V_mem) + static_cast<int16_t>(filter);
    if (sum > std::numeric_limits<int8_t>::max()) sum = std::numeric_limits<int8_t>::max();
    if (sum < std::numeric_limits<int8_t>::min()) sum = std::numeric_limits<int8_t>::min();
    return static_cast<int8_t>(sum);
}

bool PE::Comparator(int8_t new_possible_V_mem, int8_t threshold) const {
    return new_possible_V_mem >= threshold;
}

int8_t PE::VmemUpdate(int8_t new_possible_V_mem, bool spiked, int8_t reset_V_mem) const {
    return spiked ? reset_V_mem : new_possible_V_mem;
}

int8_t PE::OutputGenerator(bool spiked, int8_t timestampRegister) const {
    return spiked ? timestampRegister : kNoSpike;
}

int8_t PE::Process(int8_t timestamp, int8_t filter, int8_t threshold) {
    const int8_t new_possible_V = Accumulator(V_mem_, filter);
    spiked_ = Comparator(new_possible_V, threshold);
    V_mem_  = VmemUpdate(new_possible_V, spiked_, reset_V_mem_);
    return OutputGenerator(spiked_, timestamp);
}

// ===== PEArray implementation =====

PEArray::PEArray() : pes_{} {
    // Nothing else; PEs are default-constructed.
}

void PEArray::LatchRow(int8_t timestamp,
                       const std::array<int8_t, kNumPEs>& filter_row,
                       int8_t threshold,
                       std::uint32_t out_neuron_id) {
    latched_timestamp_   = timestamp;
    latched_threshold_   = threshold;
    latched_out_neuron_  = out_neuron_id;
    latched_filter_row_  = filter_row; // copy
    row_latched_valid_   = true;
}

bool PEArray::run() {
    if (!core_)                 return false;
    if (!row_latched_valid_)    return false;

    // Inter-stage handshake: Stage-2 may run only when valid is true.
    if (!core_->st1_st2_valid()) {
        return false; // stall; keep the latched row
    }

    bool any_progress = false;

    // Process all PEs for the current row.
    for (int i = 0; i < kNumPEs; ++i) {
        const int8_t out_ts = pes_[i].Process(
            latched_timestamp_,
            latched_filter_row_[i],
            latched_threshold_
        );

        // Resolve neuron id: prefer per-PE id if set, else use row-level id.
        const std::uint32_t nid =
            (pes_[i].output_neuron_id() != 0xFFFFFFFFu)
                ? pes_[i].output_neuron_id()
                : latched_out_neuron_;

        if (out_ts != PE::kNoSpike && nid != 0xFFFFFFFFu) {
            Entry e{};
            e.ts        = static_cast<std::uint8_t>(out_ts);
            e.neuron_id = static_cast<std::uint32_t>(nid);
            // For modeling simplicity, ignore failure (e.g., if Stage1 gate flips unexpectedly).
            (void)core_->ts_picker().Stage2Write(e);
        }
    }

    // Consume the row latch after processing all PEs.
    row_latched_valid_ = false;
    any_progress = true;
    return any_progress;
}

} // namespace sf
