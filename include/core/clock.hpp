#pragma once
// All comments are in English.
#include <functional>
#include <array>
#include <vector>
#include "common/constants.hpp"
#include "common/entry.hpp"
#include "core/core_iface.hpp"
#include "arch/output_queue.hpp"            // S0
#include "arch/smallest_ts_picker.hpp"      // S1
#include "arch/pe_array.hpp"                // S2
#include "arch/input_weight_provider.hpp"   // S3
#include "arch/intermediate_fifo.hpp"       // S3/S4
#include "arch/input_spine_buffer.hpp"      // S4 source + S5 owner
#include "arch/min_finder_batch.hpp"        // S4
#include "arch/filter_buffer.hpp"           // S3
#include "driver/weight_lut.hpp"            // S3

namespace sf {
namespace dram { class DramFormat; }

class ClockCore final : public CoreIface {
public:
    explicit ClockCore(std::size_t outq_capacity = kDefaultOutputQueueCapacity);

    // Pipeline: S0 -> S1 -> S2 -> S3 -> S4 -> S5
    bool run();

    // ---- Output sink ----
    ClockCore& SetOutputSink(std::function<bool(const Entry&)> sink);
    bool SendToOutputSink(const Entry& e) override;

    // ---- Stage-1 <-> Stage-2 handshake ----
    bool st1_st2_valid() const { return st1_st2_valid_; }

    // ---- Config / params used by stages (unchanged from before) ----
    driver::WeightLUT&       lut()             { return lut_; }
    const driver::WeightLUT& lut()       const { return lut_; }
    FilterBuffer&            filter_buffer()   { return filter_buf_; }
    const FilterBuffer&      filter_buffer() const { return filter_buf_; }
    void     SetThreshold(int8_t thr)          { threshold_ = thr; }
    int8_t   threshold() const                 { return threshold_; }
    void     SetTilesPerStep(uint16_t n)       { tiles_per_step_ = (n == 0 ? 1 : n); }
    uint16_t tiles_per_step() const            { return tiles_per_step_; }

    // ---- Components accessors ----
    OutputQueue&         output_queue()       { return out_q_; }
    SmallestTsPicker&    ts_picker()          { return ts_picker_; }
    PEArray&             pe_array()           { return pe_array_; }
    InputWeightProvider& input_weight_provider() { return iwp_; }
    MinFinderBatch&      min_finder()         { return min_finder_; }
    InputSpineBuffer&    input_spine_buffer() { return in_buf_; }

    // ---- Batch/FIFO context for S3/S4 ----
    std::array<IntermediateFIFO, kMaxBatches>&       fifos()       { return fifos_; }
    const std::array<IntermediateFIFO, kMaxBatches>& fifos() const { return fifos_; }
    void SetBatchesNeeded(int n) { batches_needed_ = (n < 1 ? 1 : (n > kMaxBatches ? kMaxBatches : n)); }
    int  batches_needed() const  { return batches_needed_; }
    void SetTotallyDrained(int b, bool v) { if (b>=0 && b<kMaxBatches) totally_drained_[b] = v; }
    bool totally_drained(int b) const { return (b>=0 && b<kMaxBatches) ? totally_drained_[b] : true; }
    int  load_batch_cursor() const { return load_batch_cursor_; }
    bool input_drained(int b) const { return (b>=0 && b<kMaxBatches) ? input_drained_[b] : false; }

    // ---- DRAM segment fetcher for S5 (PSB) ----
    // Returns true if a new line is available; fills 'out_line' bytes and provides the DramFormat.
    using DramFetchFn = std::function<bool(int batch, int spine,
                                           std::vector<std::uint8_t>& out_line,
                                           const sf::dram::DramFormat*& out_fmt)>;
    ClockCore& SetDramFetcher(DramFetchFn fn) { dram_fetcher_ = std::move(fn); return *this; }
    bool FetchNextSpineSegment(int batch, int spine,
                               std::vector<std::uint8_t>& out_line,
                               const sf::dram::DramFormat*& out_fmt) {
        return dram_fetcher_ ? dram_fetcher_(batch, spine, out_line, out_fmt) : false;
    }

private:
    // Only SmallestTsPicker modifies st1_st2_valid_
    void SetSt1St2Valid(bool v) { st1_st2_valid_ = v; }
    friend class SmallestTsPicker;

    // Only MinFinderBatch modifies these
    void AdvanceLoadBatchCursor() { ++load_batch_cursor_; }
    void SetInputDrained(int b, bool v) { if (b>=0 && b<kMaxBatches) input_drained_[b] = v; }
    friend class MinFinderBatch;

private:
    // Components
    OutputQueue         out_q_;
    SmallestTsPicker    ts_picker_;
    PEArray             pe_array_;
    InputWeightProvider iwp_;
    MinFinderBatch      min_finder_;
    InputSpineBuffer    in_buf_;

    // Handshake
    bool st1_st2_valid_ = true;

    // S3/S4 shared state
    std::array<IntermediateFIFO, kMaxBatches> fifos_{};
    std::array<bool, kMaxBatches>             totally_drained_{};
    int                                       batches_needed_ = 1;

    // Stage-4 control
    std::array<bool, kMaxBatches> input_drained_{};
    int load_batch_cursor_ = 0;

    // S2/S3 params and data
    FilterBuffer        filter_buf_;
    driver::WeightLUT   lut_;
    int8_t              threshold_ = 0;
    uint16_t            tiles_per_step_ = 1;

    // DRAM fetcher
    DramFetchFn         dram_fetcher_{};

    // Output sink
    std::function<bool(const Entry&)> out_sink_;
};

} // namespace sf
