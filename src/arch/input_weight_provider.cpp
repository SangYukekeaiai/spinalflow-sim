#include "arch/input_weight_provider.hpp"

// All comments are in English.
#include <array>
#include <optional>
#include "core/clock.hpp"                 // Core accessors
#include "arch/intermediate_fifo.hpp"     // FIFO API
#include "arch/global_merger.hpp"         // GlobalMerger::PickAndPop
#include "arch/filter_buffer.hpp"         // FilterBuffer::Row
#include "arch/pe_array.hpp"              // PEArray::LatchRow
#include "driver/weight_lut.hpp"          // WeightLUT

namespace sf {

bool InputWeightProvider::CanRunGlobalMergerThisStep() const {
    if (!core_) return false;
    const int batches = core_->batches_needed();
    bool any_active = false;

    for (int b = 0; b < batches; ++b) {
        if (core_->totally_drained(b)) continue; // ignore finished batches
        any_active = true;
        const auto& fifo = core_->fifos()[b];
        if (fifo.empty()) return false;          // requires primed head
    }
    return any_active;
}

bool InputWeightProvider::TryLatchPendingToPEArray() {
    if (!core_ || !pending_row_valid_) return false;
    auto& pea = core_->pe_array();
    if (pea.has_latch()) return false;

    // Use layer threshold from Core; neuron_id is row-level mapping for outputs.
    const int8_t thr = core_->threshold();

    pea.LatchRow(
        pending_.ts,
        reinterpret_cast<const std::array<int8_t, 128>&>(pending_.row),
        thr,
        pending_.neuron_id
    );

    pending_row_valid_ = false;
    return true;
}

bool InputWeightProvider::run() {
    if (!core_) return false;

    bool progressed = false;

    // 0) If we already have a pending row, try to hand it to PEArray first.
    if (pending_row_valid_) {
        if (TryLatchPendingToPEArray()) {
            progressed = true;
        }
        // Even if we couldn't latch now, we must not pick a new row yet.
        return progressed;
    }

    // 1) Only proceed if PEArray can accept a new row (no current latch).
    if (core_->pe_array().has_latch()) {
        return false; // backpressure from S2
    }

    // 2) Check if GlobalMerger may run this step.
    if (!CanRunGlobalMergerThisStep()) {
        return false;
    }

    // 3) Build FIFO pointer array for GlobalMerger.
    std::array<IntermediateFIFO*, kMaxBatches> refs{};
    for (int b = 0; b < core_->batches_needed(); ++b) {
        refs[b] = const_cast<IntermediateFIFO*>(&(core_->fifos()[b]));
    }

    // 4) Pick and pop the globally smallest entry.
    auto picked = GlobalMerger::PickAndPop(refs);
    if (!picked.has_value()) {
        return false;
    }

    const auto& res = *picked;
    const Entry& e  = res.entry;

    // 5) Map (neuron_id, out_tile) -> FilterBuffer row id.
    auto& lut = core_->lut();
    const uint32_t row_id = lut.RowIdFromNeuron(e.neuron_id, cur_out_tile_);

    // 6) Read the weight row from FilterBuffer.
    FilterBuffer::Row row{};
    if (!core_->filter_buffer().ReadRow(static_cast<int>(row_id), row)) {
        // Failed to read weights: treat as no progress (modeling simplicity).
        return false;
    }

    // 7) Fill the pending row latch.
    for (int i = 0; i < 128; ++i) pending_.row[i] = row[i];
    pending_.ts        = static_cast<int8_t>(e.ts);
    pending_.neuron_id = e.neuron_id;
    pending_row_valid_ = true;
    progressed = true;

    // 8) Attempt immediate hand-off to PEArray (opportunistic).
    if (TryLatchPendingToPEArray()) {
        progressed = true;
    }

    // 9) Advance the tile pointer.
    const std::uint16_t tiles_per_step = core_->tiles_per_step();
    const std::uint16_t out_tiles      = core_->lut().OutTiles();
    if (out_tiles != 0) {
        cur_out_tile_ = static_cast<std::uint16_t>(
            (cur_out_tile_ + tiles_per_step) % out_tiles
        );
    }

    return progressed;
}

} // namespace sf
