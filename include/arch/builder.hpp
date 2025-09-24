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
#include "utils/latency_stats.hpp"

namespace sf {

/**
 * Builder
 * Simplified pipeline assuming:
 *  - Each (batch, spine) is read from DRAM exactly once.
 *  - After read, the spine is immediately swapped to ACTIVE and drained into that batch's FIFO.
 *  - GlobalMerger may pop only when every non-drained batch has at least one FIFO entry.
 *
 * Stages per step:
 *   (0) Drain OutputQueue to sink
 *   (1) Move smallest-ts spikes from picker -> OutputQueue
 *   (2) Process PEs with the last latched row, push spikes to picker
 *   (3) GlobalMerger + LUT row select + latch
 *   (4) Drain current-load batch from InputSpineBuffer -> its FIFO (one item)
 *   (5) Load one (batch, spine) from DRAM into shadow and swap to ACTIVE
 */
class Builder {
public:
    struct LayerConfig {
        uint16_t inC = 0;
        uint16_t outC = 0;
        uint8_t  kH  = 0;
        uint8_t  kW  = 0;
        int8_t   threshold = 0;
        uint16_t tiles_per_step = 1;
    };

    // New DRAM reader: stream the entire physical spine for (batch, spine) once.
    // Returns number of BYTES written into dst (0 means nothing this cycle).
    using DramReadFn   = std::function<std::size_t(int batchIdx,
                                                   int physSpineId,
                                                   std::uint8_t* dst,
                                                   std::size_t   maxBytes)>;
    using OutputSinkFn = std::function<bool(const Entry&)>;

public:
    Builder(InputSpineBuffer&      in_buf,
            FilterBuffer&          filter_buf,
            std::vector<PE>&       pes_128,
            driver::BatchSpineMap& bmap,
            driver::WeightLUT&     lut);

    void ConfigureLayer(const LayerConfig& cfg);

    void SetDramReader(DramReadFn fn)   { dram_read_ = std::move(fn); }
    void SetOutputSink(OutputSinkFn fn) { out_sink_  = std::move(fn); }

    void PrefillWeightsOnce(const int8_t* base, std::size_t bytes);
    // Optional one-shot warm-up: try loading some spines of the first batch.
    void PrefillInputSpinesOnce(bool try_immediate_swap = true);

    // Run one hardware step; returns true if any stage made progress.
    bool Step();

    // Introspection
    int  NumBatchesNeeded()    const { return batches_needed_; }
    int  RequiredActiveFifos() const { return required_active_fifos_; }
    int  LoadBatchCursor()     const { return load_batch_cursor_; }
    const IntermediateFIFO& FifoOf(int b) const { return fifos_.at(b); }
    const OutputQueue&      OutputQ()          const { return out_q_; }
    std::string DebugString() const;

    // Latency stats
    const util::LatencyStats& LatencySnapshot() const;
    void ResetLatencyStats();

    void SetDRAMToInBufLatency(std::uint32_t cycles) {
        dram_fill_latency_cycles_ = cycles;
    }

private:
    // Pipeline stages
    bool Stage0_DrainOutputQueue();
    bool Stage1_SelectSmallestTimestamp();
    bool Stage2_ProcessPEs();
    bool Stage3_GlobalMergeAndFilter();
    bool Stage4_DrainCurrentBatchToFIFO();   // was MinFinderAcrossBatches
    bool Stage5_LoadNextBatchSpine();        // was refill/swap

    // Helpers
    void RecomputeBatching();
    bool CanRunGlobalMergerThisStep() const;
    int  CountPrimedFifos() const;
    bool LoadSpineFromDRAM(int batch_idx, int spine_idx, bool swap_to_active);

    // Drained-state maintenance
    bool BatchTotallyDrained(int b) const;
    void UpdateDrainStatus();

private:
    // External modules
    InputSpineBuffer&      in_buf_;
    FilterBuffer&          filter_;
    std::vector<PE>&       pes_;
    driver::BatchSpineMap& bmap_;
    driver::WeightLUT&     lut_;

    // Configuration
    LayerConfig cfg_{};

    // Runtime
    int  batches_needed_        = 1;
    int  required_active_fifos_ = 1;
    std::array<IntermediateFIFO, kMaxBatches> fifos_{};

    SmallestTsPicker ts_picker_{};
    OutputQueue      out_q_{kDefaultOutputQueueCapacity};
    uint16_t         cur_out_tile_ = 0;

    // Latched row & metadata
    bool              row_latched_valid_ = false;
    FilterBuffer::Row latched_row_{};
    int8_t            latched_timestamp_  = 0;
    int               latched_out_neuron_ = -1;

    bool              pending_row_valid_ = false;
    FilterBuffer::Row pending_row_{};
    int8_t            pending_timestamp_  = 0;
    int               pending_out_neuron_ = -1;

    // MinFinder
    MinFinderBatch min_finder_;

    // Batch loading cursor: which batch we are currently loading/draining into its FIFO
    int load_batch_cursor_ = 0;

    // Per (batch, spine) read-once flags; true after DRAM read + swap
    std::array<std::array<bool, kNumSpines>, kMaxBatches> fetched_{};
    // True if all input of batch b has been moved from InputSpineBuffer into FIFO[b]
    std::array<bool, kMaxBatches> input_drained_{};
    // True if batch b is completely finished (inputs drained and FIFO empty)
    std::array<bool, kMaxBatches> totally_drained_{};

    // I/O callbacks
    DramReadFn   dram_read_;
    OutputSinkFn out_sink_;

    // Stats
    std::uint64_t      cycle_ = 0;
    util::LatencyStats latency_{};
    std::array<std::deque<std::uint64_t>, kMaxBatches> fifo_enqueue_cycles_{};
    std::deque<std::uint64_t>                           out_enqueue_cycles_;
    // DRAM->InputSpine extra latency in cycles when Stage5 fires.
    // Keep it simple and constant per event; model as a bulk "stall".
    std::uint32_t dram_fill_latency_cycles_ = 200; // default; tune per platform
};

} // namespace sf
