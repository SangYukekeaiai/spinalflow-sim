#pragma once
#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "common/constants.hpp"
#include "common/entry.hpp"
#include "arch/input_spine_buffer.hpp"
#include "arch/intermediate_fifo.hpp"
#include "arch/min_finder_batch.hpp"
#include "arch/global_merger.hpp"
#include "arch/filter_buffer.hpp"
#include "arch/pe.hpp"
#include "driver/batch_spine_map.hpp"
#include "driver/weight_lut.hpp"

namespace sf {

/**
 * Builder
 * A hardware-like 6-stage stepper:
 *   (0) optional per-layer prefill
 *   (1) Process PEs (consume last latched row)
 *   (2) GlobalMerger + WeightLUT row select + latch
 *   (3) Intermediate FIFO bookkeeping (placeholder)
 *   (4) MinFinder: drain ONLY the active batch into its FIFO
 *   (5) InputSpineBuffer: atomic shadow->active swap for a whole batch
 *   (6) DRAM refill for the prep batch (next batch to become active)
 *
 * Batch/FIFO discipline:
 *   - One IntermediateFIFO per batch (up to 4).
 *   - Only the ACTIVE batch is drained by MinFinder into its matching FIFO.
 *   - Atomic swap requires all 16 spines' SHADOW banks to hold slices of prep_batch_.
 *   - GlobalMerger is gated only by batches that are NOT totally drained.
 */
class Builder {
public:
    struct LayerConfig {
        uint16_t inC = 0;
        uint16_t outC = 0;
        uint8_t  kH  = 0;
        uint8_t  kW  = 0;
        int8_t   threshold = 0;
        uint16_t tiles_per_step = 1;  // typically 1
    };

    // Optional DRAM reader callback. Should read 'bytes' from 'addr' into 'dst'.
    using DramReadFn = std::function<bool(driver::BatchSpineMap::Addr addr,
                                          std::uint8_t* dst,
                                          std::size_t   bytes)>;

public:
    Builder(InputSpineBuffer&      in_buf,
            FilterBuffer&          filter_buf,
            std::vector<PE>&       pes_128,
            driver::BatchSpineMap& bmap,
            driver::WeightLUT&     lut);

    // Configure one convolution layer, rebuild LUT, reset runtime states.
    void ConfigureLayer(const LayerConfig& cfg);

    // Optional: set a DRAM reader for automatic prefill/refill.
    void SetDramReader(DramReadFn fn) { dram_read_ = std::move(fn); }

    // Optional once-per-layer helpers
    void PrefillWeightsOnce(const int8_t* base, std::size_t bytes);
    void PrefillInputSpinesOnce(bool try_atomic_swap = true);

    // Run one hardware step; returns number of spikes produced by PEs this step.
    int Step();

    // Introspection / debug
    int  NumBatchesNeeded()    const { return batches_needed_; }
    int  RequiredActiveFifos() const { return required_active_fifos_; }
    int  ActiveBatch()         const { return active_batch_; }
    int  PrepBatch()           const { return prep_batch_; }
    const IntermediateFIFO& FifoOf(int b) const { return fifos_.at(b); }
    std::string DebugString() const;

private:
    // Pipeline stages
    int  Stage1_ProcessPEs();
    bool Stage2_GlobalMergeAndFilter();
    void Stage3_ProcessIFOs();
    void Stage4_MinFinderAcrossBatches();
    void Stage5_InputSpineBankSwap();
    void Stage6_CheckAndRefillFromDRAM();

    // Utilities
    void RecomputeBatching();
    bool CanRunGlobalMergerThisStep() const;  // gate based on non-drained batches
    int  CountPrimedFifos() const;            // debug helper (not used for gating)

    // Drained-state maintenance
    bool BatchTotallyDrained(int b) const;
    void UpdateDrainStatus();

private:
    // External modules (by reference)
    InputSpineBuffer&      in_buf_;
    FilterBuffer&          filter_;
    std::vector<PE>&       pes_;
    driver::BatchSpineMap& bmap_;
    driver::WeightLUT&     lut_;

    // Configuration
    LayerConfig cfg_{};

    // Runtime state
    int  batches_needed_        = 1;  // #batches used (cap at 4)
    int  required_active_fifos_ = 1;  // usually equals batches_needed_
    std::array<IntermediateFIFO, 4> fifos_{};  // one FIFO per batch

    // Per-(batch, spine) DRAM address cursor
    std::array<std::array<int, kNumSpines>, kMaxBatches> next_addr_idx_{};

    // Output tile pointer (round-robin)
    uint16_t cur_out_tile_ = 0;

    // Latched filter row for next PE processing
    bool              row_latched_valid_ = false;
    FilterBuffer::Row latched_row_{};
    int8_t            latched_timestamp_ = 0;

    // MinFinder bound to input buffer
    MinFinderBatch    min_finder_;

    // Batch scheduling across the 16 spines
    int active_batch_ = -1;                         // current ACTIVE batch id; -1 = none
    int prep_batch_   = 0;                          // the batch being prepared in SHADOW
    std::array<int, kNumSpines> shadow_owner_{};    // per-spine SHADOW owner batch (-1 if empty)

    // Permanently exhausted batches
    std::array<bool, 4> totally_drained_{};         // monotonic true once drained

    // Optional DRAM reader
    DramReadFn        dram_read_;
};

} // namespace sf
