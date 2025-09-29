#pragma once
// All comments are in English.

#include <cstdint>
#include <array>
#include "common/constants.hpp"    // kMaxBatches, etc.

namespace sf {

// Forward declarations to keep the header light.
class ClockCore;
class IntermediateFIFO;
class FilterBuffer;
namespace driver { class WeightLUT; }

/**
 * InputWeightProvider (Stage-3)
 *
 * Responsibilities:
 *  - Gate GlobalMerger execution: only when every non-drained batch FIFO is primed.
 *  - Pick one entry via GlobalMerger and map (neuron_id, out_tile) -> row_id using WeightLUT.
 *  - Read the 128-lane weight row from FilterBuffer.
 *  - Hold a pending row latch (row + ts + neuron_id) until PEArray can accept it.
 *  - Advance the tile pointer per step (tiles_per_step mod OutTiles()).
 *
 * Backpressure & ordering:
 *  - Do NOT pick a new entry if a pending row exists or PEArray already has a latch.
 *  - Only transfer the pending row to PEArray when PEArray.has_latch() == false.
 */
class InputWeightProvider {
public:
    InputWeightProvider() = default;

    // Wire to Core to access FIFOs, LUT, FilterBuffer, PEArray, and params.
    void RegisterCore(ClockCore* core) { core_ = core; }

    // Execute one Stage-3 attempt.
    // Returns true if any progress was made (picked, read, or latched to PEArray).
    bool run();

private:
    // Check if GlobalMerger may run this step:
    // every NOT-YET-DRAINED batch must have a primed FIFO head.
    bool CanRunGlobalMergerThisStep() const;

    // Try to hand the pending row to PEArray if possible.
    bool TryLatchPendingToPEArray();

private:
    ClockCore* core_ = nullptr;  // not owned

    // Pending row latch to hand off to PEArray.
    bool pending_row_valid_ = false;
    // Stored as FilterBuffer::Row; defined in cpp to avoid including the type here.
    struct PendingRow {
        std::array<int8_t, 128> row;    // matches FilterBuffer::Row width
        int8_t                  ts = 0;
        std::uint32_t           neuron_id = 0xFFFFFFFFu;
    } pending_{};

    // Tile pointer (local to Stage-3).
    std::uint16_t cur_out_tile_ = 0;
};

} // namespace sf
