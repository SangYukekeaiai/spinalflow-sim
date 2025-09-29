#include "arch/builder.hpp"
#include <algorithm>
#include <sstream>
#include "utils/latency_stats.hpp"

namespace sf {

// ===== ctor / config =====

Builder::Builder(InputSpineBuffer&      in_buf,
                 FilterBuffer&          filter_buf,
                 std::vector<PE>&       pes_128,
                 driver::BatchSpineMap& bmap,
                 driver::WeightLUT&     lut)
    : in_buf_(in_buf),
      filter_(filter_buf),
      pes_(pes_128),
      bmap_(bmap),
      lut_(lut),
      min_finder_(in_buf_) {
    // All comments are in English.
    if (pes_.size() != static_cast<size_t>(FilterBuffer::kNumPEs)) {
        throw std::invalid_argument("Builder requires exactly 128 PEs.");
    }
    for (auto& row : fetched_)       row.fill(false);
    input_drained_.fill(false);
    totally_drained_.fill(false);
}

void Builder::ResetLatencyStats() {
    cycle_   = 0;
    latency_ = {};
    for (auto& q : fifo_enqueue_cycles_) q.clear();
    out_enqueue_cycles_.clear();
    pending_row_valid_  = false;
}

void Builder::ConfigureLayer(const LayerConfig& cfg) {
    cfg_ = cfg;

    driver::WeightLUT::Params p;
    p.inC     = cfg_.inC;
    p.outC    = cfg_.outC;
    p.kH      = cfg_.kH;
    p.kW      = cfg_.kW;
    p.peLanes = 128;
    lut_.Build(p);  // throws if invalid

    // Reset runtime state
    in_buf_.Flush();
    for (auto& f : fifos_) f.clear();
    out_q_.clear();
    ts_picker_.Clear();

    row_latched_valid_  = false;
    latched_timestamp_  = 0;
    latched_out_neuron_ = -1;
    pending_row_valid_  = false;
    cur_out_tile_       = 0;

    load_batch_cursor_ = 0;
    for (auto& row : fetched_) row.fill(false);
    input_drained_.fill(false);
    totally_drained_.fill(false);

    ResetLatencyStats();
    RecomputeBatching();
}

void Builder::PrefillWeightsOnce(const int8_t* base, std::size_t bytes) {
    if (!base || bytes == 0) return;
    filter_.LoadFromDRAM(base, bytes);
}

void Builder::PrefillInputSpinesOnce(bool try_immediate_swap) {
    if (!dram_read_) return;
    if (load_batch_cursor_ >= batches_needed_) return;

    const int b = load_batch_cursor_;
    // Try to read as many spines as possible for the current batch.
    for (int s = 0; s < kNumSpines; ++s) {
        (void)LoadSpineFromDRAM(b, s, try_immediate_swap);
    }
}

// ===== main step =====

bool Builder::Step() {
    ++cycle_;
    ++latency_.stages.steps;

    // Commit a pending row into the PE latch if available.
    if (!row_latched_valid_ && pending_row_valid_) {
        latched_row_        = pending_row_;
        latched_timestamp_  = pending_timestamp_;
        latched_out_neuron_ = pending_out_neuron_;
        row_latched_valid_  = true;
        pending_row_valid_  = false;
    }

    const bool s0 = Stage0_DrainOutputQueue();
    const bool s1 = Stage1_SelectSmallestTimestamp();
    const bool s2 = Stage2_ProcessPEs();
    const bool s3 = Stage3_GlobalMergeAndFilter();
    const bool s4 = Stage4_DrainCurrentBatchToFIFO();
    const bool s5 = Stage5_LoadNextBatchSpine();

    if (s0) ++latency_.stages.stage_hits[0];
    if (s1) ++latency_.stages.stage_hits[1];
    if (s2) ++latency_.stages.stage_hits[2];
    if (s3) ++latency_.stages.stage_hits[3];
    if (s4) ++latency_.stages.stage_hits[4];
    if (s5) ++latency_.stages.stage_hits[5];

    const bool did = s0 || s1 || s2 || s3 || s4 || s5;
    if (!did) ++latency_.stages.idle_cycles;
    if (s5) {
        // Guard against underflow if user sets latency to 0.
        const std::uint64_t extra = (dram_fill_latency_cycles_ > 0)
                                      ? (static_cast<std::uint64_t>(dram_fill_latency_cycles_) - 1ULL)
                                      : 0ULL;

        // Advance simulated time by the remaining stall cycles.
        cycle_ += extra;

        // Attribute all extra time to idle (stall) cycles.
        latency_.stages.idle_cycles += extra;

        // (Optional) If you track a dedicated counter, you can also do:
        // latency_.dram_fill_penalty += extra;  // <-- add such a field if desired
    }
    UpdateDrainStatus();
    return did;
}

// ===== stages =====

bool Builder::Stage0_DrainOutputQueue() {
    if (out_q_.empty()) return false;
    if (!out_sink_) return false;

    Entry e{};
    if (!out_q_.front(e)) return false;
    if (!out_sink_(e))    return false;
    if (!out_q_.pop(e))   return false;

    if (!out_enqueue_cycles_.empty()) {
        const std::uint64_t birth = out_enqueue_cycles_.front();
        out_enqueue_cycles_.pop_front();
        const std::uint64_t wait = (cycle_ >= birth) ? (cycle_ - birth) : 0;
        util::AccumulateQueueLatency(latency_.output_queue, wait);
    }
    return true;
}

bool Builder::Stage1_SelectSmallestTimestamp() {
    if (out_q_.full()) return false;

    bool moved = false;
    Entry e{};
    while (!out_q_.full()) {
        if (!ts_picker_.PopSmallest(e)) break;
        if (!out_q_.push_entry(e)) { ts_picker_.Push(e); break; }
        out_enqueue_cycles_.push_back(cycle_);
        moved = true;
    }
    return moved;
}

bool Builder::Stage2_ProcessPEs() {
    if (!row_latched_valid_) return false;

    for (int i = 0; i < FilterBuffer::kNumPEs; ++i) {
        const int8_t w   = latched_row_[i];
        const int8_t out = pes_[i].Process(latched_timestamp_, w, cfg_.threshold);
        if (out != PE::kNoSpike && latched_out_neuron_ >= 0) {
            Entry e{};
            e.ts        = static_cast<uint8_t>(out);
            e.neuron_id = static_cast<uint32_t>(latched_out_neuron_);
            ts_picker_.Push(e);
        }
    }
    row_latched_valid_ = false;
    return true;
}

bool Builder::CanRunGlobalMergerThisStep() const {
    // Require every NOT-YET-DRAINED batch to have a primed FIFO head.
    bool any_active_batch = false;
    for (int b = 0; b < batches_needed_; ++b) {
        if (totally_drained_[b]) continue;
        any_active_batch = true;
        if (fifos_[b].empty()) return false;
    }
    return any_active_batch;
}


bool Builder::Stage3_GlobalMergeAndFilter() {
    if (!CanRunGlobalMergerThisStep()) return false;
    if (pending_row_valid_)           return false;

    std::array<IntermediateFIFO*, kMaxBatches> refs{};
    for (int b = 0; b < batches_needed_; ++b) refs[b] = &fifos_[b];

    auto picked = GlobalMerger::PickAndPop(refs);
    if (!picked.has_value()) return false;

    const auto& res = *picked;
    if (res.fifo_index >= 0 && res.fifo_index < kMaxBatches) {
        auto& track = fifo_enqueue_cycles_[res.fifo_index];
        if (!track.empty()) {
            const std::uint64_t birth = track.front(); track.pop_front();
            const std::uint64_t wait  = (cycle_ >= birth) ? (cycle_ - birth) : 0;
            util::AccumulateQueueLatency(latency_.fifo_wait, wait);
        }
    }

    const Entry& picked_entry = res.entry;
    const uint32_t neuron_id = picked_entry.neuron_id;
    const uint16_t out_tile  = cur_out_tile_;
    const uint32_t row_id    = lut_.RowIdFromNeuron(neuron_id, out_tile);

    FilterBuffer::Row row{};
    if (!filter_.ReadRow(static_cast<int>(row_id), row)) return false;

    pending_row_        = row;
    pending_timestamp_  = static_cast<int8_t>(picked_entry.ts);
    pending_out_neuron_ = static_cast<int>(neuron_id);
    pending_row_valid_  = true;

    cur_out_tile_ = static_cast<uint16_t>((cur_out_tile_ + cfg_.tiles_per_step) % lut_.OutTiles());
    return true;
}

bool Builder::Stage4_DrainCurrentBatchToFIFO() {
    // Drain from the current load batch into its FIFO, one item per step.
    if (load_batch_cursor_ >= batches_needed_) return false;
    const int b = load_batch_cursor_;
    if (fifos_[b].full()) return false;

    const bool moved = min_finder_.DrainOneInto(fifos_[b]);
    if (!moved) {
        // If nothing moved and all spines are empty for this batch, mark input drained and advance cursor.
        bool empty_all = true;
        for (int s = 0; s < kNumSpines; ++s) {
            if (!in_buf_.Empty(s)) { empty_all = false; break; }
        }
        if (empty_all) {
            input_drained_[b] = true;
            load_batch_cursor_++; // proceed to next batch
        }
        return false;
    }
    fifo_enqueue_cycles_[b].push_back(cycle_);
    return true;
}

bool Builder::Stage5_LoadNextBatchSpine() {
    if (!dram_read_) return false;
    if (load_batch_cursor_ >= batches_needed_) return false;

    const int b = load_batch_cursor_;

    // Load at most one spine per cycle to model limited bandwidth.
    for (int s = 0; s < kNumSpines; ++s) {
        if (LoadSpineFromDRAM(b, s, true)) {
            return true;
        }
    }
    return false;
}

// ===== drained-state maintenance & utilities =====

void Builder::RecomputeBatching() {
    batches_needed_        = std::max(1, std::min(bmap_.NumBatches(), kMaxBatches));
    required_active_fifos_ = batches_needed_;
}

bool Builder::LoadSpineFromDRAM(int batch_idx, int spine_idx, bool swap_to_active) {
    if (!dram_read_) return false;
    if (batch_idx < 0 || batch_idx >= batches_needed_) return false;
    if (spine_idx < 0 || spine_idx >= kNumSpines) return false;
    if (fetched_[batch_idx][spine_idx]) return false;

    std::array<std::uint8_t, kCapacityPerSpine * sizeof(Entry)> tmp{};
    const std::size_t got = dram_read_(batch_idx, spine_idx, tmp.data(), tmp.size());
    if (got == 0) return false;
    if (got % sizeof(Entry)) throw std::runtime_error("DRAM read misaligned");

    in_buf_.LoadSpineShadowFromDRAM(spine_idx, tmp.data(), got);
    if (swap_to_active) (void)in_buf_.SwapToShadow(spine_idx);
    fetched_[batch_idx][spine_idx] = true;
    return true;
}

bool Builder::BatchTotallyDrained(int b) const {
    // 1) All spines fetched once
    for (int s = 0; s < kNumSpines; ++s) {
        if (!fetched_[b][s]) return false;
    }
    // 2) Input of this batch has been completely moved into its FIFO
    if (!input_drained_[b]) return false;
    // 3) FIFO is empty
    if (!fifos_[b].empty()) return false;
    return true;
}

void Builder::UpdateDrainStatus() {
    for (int b = 0; b < batches_needed_; ++b) {
        if (totally_drained_[b]) continue;
        if (BatchTotallyDrained(b)) {
            totally_drained_[b] = true;
        }
    }
}

std::string Builder::DebugString() const {
    std::ostringstream oss;
    oss << "Builder{"
        << "inC=" << cfg_.inC << ", outC=" << cfg_.outC
        << ", kH=" << int(cfg_.kH) << ", kW=" << int(cfg_.kW)
        << ", tiles=" << lut_.OutTiles()
        << ", batches_needed=" << batches_needed_
        << ", required_active_fifos=" << required_active_fifos_
        << ", row_latched=" << (row_latched_valid_ ? "Y" : "N")
        << ", cur_out_tile=" << cur_out_tile_
        << ", load_batch_cursor=" << load_batch_cursor_
        << ", outq_size=" << out_q_.size()
        << ", drained=[";
    for (int b = 0; b < batches_needed_; ++b) {
        oss << (totally_drained_[b] ? '1' : '0');
        if (b + 1 < batches_needed_) oss << ",";
    }
    oss << "]}";
    return oss.str();
}

const util::LatencyStats& Builder::LatencySnapshot() const {
    return latency_;
}

} // namespace sf
