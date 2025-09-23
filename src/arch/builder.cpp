#include "arch/builder.hpp"
#include <algorithm>
#include <cstring>
#include <sstream>
#include "utils/latency_stats.hpp"  // ensure TU sees util helpers

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
    for (auto& v : next_addr_idx_) v.fill(0);
    shadow_owner_.fill(-1);
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
    for (auto& v : next_addr_idx_) v.fill(0);

    row_latched_valid_  = false;
    latched_timestamp_  = 0;
    latched_out_neuron_ = -1;
    pending_row_valid_  = false;
    cur_out_tile_       = 0;

    active_batch_ = -1;
    prep_batch_   = 0;
    shadow_owner_.fill(-1);
    totally_drained_.fill(false);

    ResetLatencyStats();
    RecomputeBatching();
}

void Builder::PrefillWeightsOnce(const int8_t* base, std::size_t bytes) {
    if (!base || bytes == 0) return;
    filter_.LoadFromDRAM(base, bytes);
}

void Builder::PrefillInputSpinesOnce(bool try_atomic_swap) {
    if (!dram_read_) return;

    const int b = prep_batch_;
    for (int s = 0; s < kNumSpines; ++s) {
        const auto& addrs = bmap_.Get(b, s);
        int& idx          = next_addr_idx_[b][s];
        if (idx >= static_cast<int>(addrs.size())) continue;

        std::array<std::uint8_t, kCapacityPerSpine * sizeof(Entry)> tmp{};
        if (dram_read_(addrs[idx], tmp.data(), tmp.size())) {
            in_buf_.LoadSpineShadowFromDRAM(s, tmp.data(), tmp.size());
            shadow_owner_[s] = b;
            idx += 1;
        }
    }

    if (try_atomic_swap) {
        bool ready = true;
        for (int s = 0; s < kNumSpines; ++s) {
            if (shadow_owner_[s] != b) { ready = false; break; }
        }
        if (ready) {
            for (int s = 0; s < kNumSpines; ++s) {
                (void)in_buf_.SwapToShadow(s);
                shadow_owner_[s] = -1;
            }
            active_batch_ = b;
            prep_batch_   = (batches_needed_ == 1) ? 0 : (b + 1) % batches_needed_;
        }
    }
}

// ===== main step =====

bool Builder::Step() {
    ++cycle_;
    ++latency_.stages.steps;

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
    const bool s4 = Stage4_MinFinderAcrossBatches();
    const bool s5 = Stage5_InputSpineBankSwap();
    const bool s6 = Stage6_CheckAndRefillFromDRAM();

    if (s0) ++latency_.stages.stage_hits[0];
    if (s1) ++latency_.stages.stage_hits[1];
    if (s2) ++latency_.stages.stage_hits[2];
    if (s3) ++latency_.stages.stage_hits[3];
    if (s4) ++latency_.stages.stage_hits[4];
    if (s5) ++latency_.stages.stage_hits[5];
    if (s6) ++latency_.stages.stage_hits[6];

    const bool did = s0 || s1 || s2 || s3 || s4 || s5 || s6;
    if (!did) ++latency_.stages.idle_cycles;

    UpdateDrainStatus();
    return did;
}

// ===== stages =====

bool Builder::Stage0_DrainOutputQueue() {
    if (out_q_.empty()) return false;
    if (!out_sink_) return false;

    Entry e{};
    if (!out_q_.front(e)) return false;
    if (!out_sink_(e)) return false;

    if (!out_q_.pop(e)) return false;

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
    Entry next{};

    while (!out_q_.full()) {
        if (!ts_picker_.PopSmallest(next)) {
            break;
        }

        if (!out_q_.push_entry(next)) {
            // Failed to enqueue; reinsert and stop to avoid losing the entry.
            ts_picker_.Push(next);
            break;
        }

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
    for (int b = 0; b < batches_needed_; ++b) {
        if (totally_drained_[b]) continue;
        if (fifos_[b].empty()) return false;
    }
    return true;
}

int Builder::CountPrimedFifos() const {
    int n = 0;
    for (int b = 0; b < batches_needed_; ++b) {
        if (!fifos_[b].empty()) ++n;
    }
    return n;
}

bool Builder::Stage3_GlobalMergeAndFilter() {
    if (!CanRunGlobalMergerThisStep()) {
        return false;
    }
    if (pending_row_valid_) {
        return false;
    }

    std::array<IntermediateFIFO*, 4> refs{};
    for (int b = 0; b < batches_needed_; ++b) refs[b] = &fifos_[b];

    auto picked = GlobalMerger::PickAndPop(refs);
    if (!picked.has_value()) return false;

    const auto& res = *picked;
    if (res.fifo_index >= 0 && res.fifo_index < kMaxBatches) {
        auto& tracking = fifo_enqueue_cycles_[res.fifo_index];
        if (!tracking.empty()) {
            const std::uint64_t birth = tracking.front();
            tracking.pop_front();
            const std::uint64_t wait = (cycle_ >= birth) ? (cycle_ - birth) : 0;
            util::AccumulateQueueLatency(latency_.fifo_wait, wait);
        }
    }

    const Entry& picked_entry = res.entry;
    const uint32_t neuron_id = picked_entry.neuron_id;
    const uint16_t out_tile  = cur_out_tile_;
    const uint32_t row_id    = lut_.RowIdFromNeuron(neuron_id, out_tile);

    FilterBuffer::Row row{};
    if (!filter_.ReadRow(static_cast<int>(row_id), row)) return false;

    pending_row_         = row;
    pending_timestamp_   = static_cast<int8_t>(picked_entry.ts);
    pending_out_neuron_  = static_cast<int>(neuron_id);
    pending_row_valid_   = true;

    cur_out_tile_ = static_cast<uint16_t>((cur_out_tile_ + cfg_.tiles_per_step) % lut_.OutTiles());
    return true;
}

bool Builder::Stage4_MinFinderAcrossBatches() {
    if (active_batch_ < 0) return false;
    if (totally_drained_[active_batch_]) return false;
    if (fifos_[active_batch_].full()) return false;

    const bool moved = min_finder_.DrainOneInto(fifos_[active_batch_]);
    if (!moved) return false;

    fifo_enqueue_cycles_[active_batch_].push_back(cycle_);
    return true;
}

bool Builder::Stage5_InputSpineBankSwap() {
    if (totally_drained_[prep_batch_]) {
        return false;
    }

    bool ready = true;
    for (int s = 0; s < kNumSpines; ++s) {
        if (shadow_owner_[s] != prep_batch_) { ready = false; break; }
    }
    if (!ready) return false;

    for (int s = 0; s < kNumSpines; ++s) {
        (void)in_buf_.SwapToShadow(s);
        shadow_owner_[s] = -1;
    }
    active_batch_ = prep_batch_;
    prep_batch_   = (batches_needed_ == 1) ? 0 : (prep_batch_ + 1) % batches_needed_;
    return true;
}

bool Builder::Stage6_CheckAndRefillFromDRAM() {
    if (!dram_read_) return false;

    const int b = prep_batch_;
    if (b < 0 || totally_drained_[b]) return false;

    bool did = false;
    for (int s = 0; s < kNumSpines; ++s) {
        if (shadow_owner_[s] == b) continue;

        const auto& addrs = bmap_.Get(b, s);
        int& idx          = next_addr_idx_[b][s];
        if (idx >= static_cast<int>(addrs.size())) {
            continue;
        }

        std::array<std::uint8_t, kCapacityPerSpine * sizeof(Entry)> tmp{};
        if (dram_read_(addrs[idx], tmp.data(), tmp.size())) {
            in_buf_.LoadSpineShadowFromDRAM(s, tmp.data(), tmp.size());
            shadow_owner_[s] = b;
            idx += 1;
            did = true;
            break;  // model one-spine-per-cycle refill
        }
    }
    return did;
}

// ===== drained-state maintenance & utilities =====

void Builder::RecomputeBatching() {
    batches_needed_        = std::max(1, std::min(bmap_.NumBatches(), kMaxBatches));
    required_active_fifos_ = batches_needed_;
}

bool Builder::BatchTotallyDrained(int b) const {
    for (int s = 0; s < kNumSpines; ++s) {
        const auto& addrs = bmap_.Get(b, s);
        if (next_addr_idx_[b][s] < static_cast<int>(addrs.size())) {
            return false;
        }
    }
    for (int s = 0; s < kNumSpines; ++s) {
        if (shadow_owner_[s] == b) return false;
    }
    if (active_batch_ == b) {
        for (int s = 0; s < kNumSpines; ++s) {
            if (!in_buf_.Empty(s)) return false;
        }
    }
    if (!fifos_[b].empty()) return false;

    return true;
}

void Builder::UpdateDrainStatus() {
    for (int b = 0; b < batches_needed_; ++b) {
        if (totally_drained_[b]) continue;
        if (BatchTotallyDrained(b)) {
            totally_drained_[b] = true;

            if (active_batch_ == b) active_batch_ = -1;

            if (prep_batch_ == b) {
                for (int step = 0; step < batches_needed_; ++step) {
                    int cand = (prep_batch_ + 1 + step) % batches_needed_;
                    if (!totally_drained_[cand]) { prep_batch_ = cand; break; }
                }
            }
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
        << ", primed=" << CountPrimedFifos()
        << ", row_latched=" << (row_latched_valid_ ? "Y" : "N")
        << ", cur_out_tile=" << cur_out_tile_
        << ", active_batch=" << active_batch_
        << ", prep_batch="   << prep_batch_
        << ", outq_size="    << out_q_.size()
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
