#pragma once
#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>
#include <deque>

#include "common/constants.hpp"
#include "common/entry.hpp"
#include "arch/input_spine_buffer.hpp"
#include "arch/intermediate_fifo.hpp"
#include "arch/min_finder_batch.hpp"
#include "arch/smallest_ts_picker.hpp"
#include "arch/global_merger.hpp"
#include "arch/filter_buffer.hpp"
#include "arch/pe.hpp"
#include "arch/output_queue.hpp"
#include "driver/batch_spine_map.hpp"
#include "driver/weight_lut.hpp"
#include "utils/latency_stats.hpp"   // <-- moved stats types here

namespace sf {

/**
 * Builder
 * A hardware-like multi-stage stepper:
 *   (0) Drain OutputQueue to an optional sink (e.g., DRAM writer)
 *   (1) Select the smallest-ts PE spike and enqueue to the OutputQueue
 *   (2) Process PEs using last latched filter row; collect spikes for the picker
 *   (3) GlobalMerger + WeightLUT row select + latch (provides neuron id & timestamp)
 *   (4) MinFinder: drain ONLY the active batch into its FIFO
 *   (5) InputSpineBuffer: atomic shadow->active swap for a whole batch
 *   (6) DRAM refill for the prep batch (next batch to become active)
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

    using DramReadFn = std::function<bool(driver::BatchSpineMap::Addr addr,
                                          std::uint8_t* dst,
                                          std::size_t   bytes)>;
    using OutputSinkFn = std::function<bool(const Entry&)>;

public:
    Builder(InputSpineBuffer&      in_buf,
            FilterBuffer&          filter_buf,
            std::vector<PE>&       pes_128,
            driver::BatchSpineMap& bmap,
            driver::WeightLUT&     lut);

    void ConfigureLayer(const LayerConfig& cfg);

    void SetDramReader(DramReadFn fn) { dram_read_ = std::move(fn); }
    void SetOutputSink(OutputSinkFn fn) { out_sink_ = std::move(fn); }

    void PrefillWeightsOnce(const int8_t* base, std::size_t bytes);
    void PrefillInputSpinesOnce(bool try_atomic_swap = true);

    // Run one hardware step; returns true iff any stage processed work this step.
    bool Step();

    // Introspection / debug
    int  NumBatchesNeeded()    const { return batches_needed_; }
    int  RequiredActiveFifos() const { return required_active_fifos_; }
    int  ActiveBatch()         const { return active_batch_; }
    int  PrepBatch()           const { return prep_batch_; }
    const IntermediateFIFO& FifoOf(int b) const { return fifos_.at(b); }
    const OutputQueue&      OutputQ() const { return out_q_; }
    std::string DebugString() const;

    // === Latency stats API (moved types to utils) ===
    const util::LatencyStats& LatencySnapshot() const;
    void ResetLatencyStats();

private:
    // Pipeline stages (all return whether they processed meaningful work)
    bool Stage0_DrainOutputQueue();
    bool Stage1_SelectSmallestTimestamp();
    bool Stage2_ProcessPEs();
    bool Stage3_GlobalMergeAndFilter();
    bool Stage4_MinFinderAcrossBatches();
    bool Stage5_InputSpineBankSwap();
    bool Stage6_CheckAndRefillFromDRAM();

    // Utilities
    void RecomputeBatching();
    bool CanRunGlobalMergerThisStep() const;
    int  CountPrimedFifos() const;

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
    int  batches_needed_        = 1;
    int  required_active_fifos_ = 1;
    std::array<IntermediateFIFO, 4> fifos_{};

    std::array<std::array<int, kNumSpines>, kMaxBatches> next_addr_idx_{};
    SmallestTsPicker ts_picker_{};
    OutputQueue out_q_{kDefaultOutputQueueCapacity};
    uint16_t    cur_out_tile_ = 0;

    // Latched row & meta
    bool              row_latched_valid_ = false;
    FilterBuffer::Row latched_row_{};
    int8_t            latched_timestamp_ = 0;
    int               latched_out_neuron_ = -1;

    bool              pending_row_valid_ = false;
    FilterBuffer::Row pending_row_{};
    int8_t            pending_timestamp_ = 0;
    int               pending_out_neuron_ = -1;

    // MinFinder
    MinFinderBatch    min_finder_;

    // Batch scheduling
    int active_batch_ = -1;
    int prep_batch_   = 0;
    std::array<int, kNumSpines> shadow_owner_{};
    std::array<bool, 4>         totally_drained_{};

    // Optional I/O callbacks
    DramReadFn   dram_read_;
    OutputSinkFn out_sink_;

    // === Stats state ===
    std::uint64_t            cycle_ = 0;
    util::LatencyStats       latency_{};
    std::array<std::deque<std::uint64_t>, kMaxBatches> fifo_enqueue_cycles_{};
    std::deque<std::uint64_t>                           out_enqueue_cycles_;
};

} // namespace sf
