#include "arch/builder.hpp"
#include <algorithm>
#include <cstring>
#include <sstream>

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
    for (auto& v : next_addr_idx_) v.fill(0);
    row_latched_valid_ = false;
    cur_out_tile_      = 0;

    active_batch_ = -1;
    prep_batch_   = 0;
    shadow_owner_.fill(-1);
    totally_drained_.fill(false);

    RecomputeBatching();
}

void Builder::PrefillWeightsOnce(const int8_t* base, std::size_t bytes) {
    if (!base || bytes == 0) return;
    filter_.LoadFromDRAM(base, bytes);
}

void Builder::PrefillInputSpinesOnce(bool try_atomic_swap) {
    // Prepare SHADOW for the initial prep_batch_ across all 16 spines.
    if (!dram_read_) return;  // caller must prefill manually if not provided

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

    // Optionally attempt an atomic swap immediately if all SHADOW banks belong to b.
    if (try_atomic_swap) {
        bool ready = true;
        for (int s = 0; s < kNumSpines; ++s) {
            if (shadow_owner_[s] != b) { ready = false; break; }
        }
        if (ready) {
            for (int s = 0; s < kNumSpines; ++s) {
                (void)in_buf_.SwapToShadow(s);
                shadow_owner_[s] = -1;  // consumed by swap
            }
            active_batch_ = b;
            prep_batch_   = (batches_needed_ == 1) ? 0 : (b + 1) % batches_needed_;
        }
    }
}

// ===== main step =====

int Builder::Step() {
    int spikes = Stage1_ProcessPEs();       // (1) PEs
    (void)Stage2_GlobalMergeAndFilter();    // (2) GlobalMerger + LUT row latch
    Stage3_ProcessIFOs();                   // (3) IFIFO bookkeeping (placeholder)
    Stage4_MinFinderAcrossBatches();        // (4) MinFinder -> ACTIVE batch only
    Stage5_InputSpineBankSwap();            // (5) Atomic bank swap if SHADOW ready
    Stage6_CheckAndRefillFromDRAM();        // (6) Refill SHADOW for prep_batch_

    UpdateDrainStatus();                    // maintain drained flags & pointers
    return spikes;
}

// ===== stages =====

int Builder::Stage1_ProcessPEs() {
    if (!row_latched_valid_) return 0;

    int produced = 0;
    for (int i = 0; i < FilterBuffer::kNumPEs; ++i) {
        const int8_t w   = latched_row_[i];
        const int8_t out = pes_[i].Process(latched_timestamp_, w, cfg_.threshold);
        if (out != PE::kNoSpike) ++produced;
    }
    row_latched_valid_ = false;  // one-cycle latch
    return produced;
}

bool Builder::CanRunGlobalMergerThisStep() const {
    // Gate by batches that are NOT totally drained:
    // every non-drained batch must have a FIFO head available this step.
    for (int b = 0; b < batches_needed_; ++b) {
        if (totally_drained_[b]) continue;     // ignore drained batches
        if (fifos_[b].empty()) return false;   // non-drained batch needs a head
    }
    return true;  // all relevant FIFOs primed (or all are drained)
}

int Builder::CountPrimedFifos() const {
    int n = 0;
    for (int b = 0; b < batches_needed_; ++b) {
        if (!fifos_[b].empty()) ++n;
    }
    return n;
}

bool Builder::Stage2_GlobalMergeAndFilter() {
    if (!CanRunGlobalMergerThisStep()) {
        // Stall between Stage 3 and Stage 2 if not all required FIFOs are primed.
        return false;
    }

    std::array<IntermediateFIFO*, 4> refs{};
    for (int b = 0; b < batches_needed_; ++b) refs[b] = &fifos_[b];

    auto picked = GlobalMerger::PickAndPop(refs);
    if (!picked.has_value()) return false;

    const uint32_t neuron_id = static_cast<uint32_t>(picked->neuron_id);
    const uint16_t out_tile  = cur_out_tile_;
    const uint32_t row_id    = lut_.RowIdFromNeuron(neuron_id, out_tile);

    FilterBuffer::Row row{};
    if (!filter_.ReadRow(static_cast<int>(row_id), row)) return false;

    latched_row_        = row;
    latched_timestamp_  = static_cast<int8_t>(picked->ts);
    row_latched_valid_  = true;

    cur_out_tile_ = static_cast<uint16_t>((cur_out_tile_ + cfg_.tiles_per_step) % lut_.OutTiles());
    return true;
}

void Builder::Stage3_ProcessIFOs() {
    // Placeholder: reserved for accounting/backpressure if needed later.
}

void Builder::Stage4_MinFinderAcrossBatches() {
    // Drain ONLY the ACTIVE batch into its dedicated FIFO.
    if (active_batch_ < 0) return;                // no batch active yet
    if (totally_drained_[active_batch_]) return;  // nothing to drain from a drained batch
    if (fifos_[active_batch_].full()) return;     // cannot accept more
    (void)min_finder_.DrainBatchInto(fifos_[active_batch_]);
}

void Builder::Stage5_InputSpineBankSwap() {
    // Atomic swap only when ALL 16 spines' SHADOW banks belong to prep_batch_.
    if (totally_drained_[prep_batch_]) {
        // Skip preparing this batch; a later UpdateDrainStatus() will advance prep.
        return;
    }

    bool ready = true;
    for (int s = 0; s < kNumSpines; ++s) {
        if (shadow_owner_[s] != prep_batch_) { ready = false; break; }
    }
    if (!ready) return;

    // Perform atomic swap across all spines; SHADOW becomes ACTIVE for this batch.
    for (int s = 0; s < kNumSpines; ++s) {
        (void)in_buf_.SwapToShadow(s);
        shadow_owner_[s] = -1;  // consumed
    }
    active_batch_ = prep_batch_;
    prep_batch_   = (batches_needed_ == 1) ? 0 : (prep_batch_ + 1) % batches_needed_;
}

void Builder::Stage6_CheckAndRefillFromDRAM() {
    if (!dram_read_) return;

    const int b = prep_batch_;  // only refill for the PREP batch
    if (b < 0 || totally_drained_[b]) return;

    for (int s = 0; s < kNumSpines; ++s) {
        if (shadow_owner_[s] == b) continue;  // already prepared for this batch

        const auto& addrs = bmap_.Get(b, s);
        int& idx          = next_addr_idx_[b][s];
        if (idx >= static_cast<int>(addrs.size())) {
            // No more slices for this (batch, spine); leave SHADOW empty for s.
            continue;
        }

        std::array<std::uint8_t, kCapacityPerSpine * sizeof(Entry)> tmp{};
        if (dram_read_(addrs[idx], tmp.data(), tmp.size())) {
            in_buf_.LoadSpineShadowFromDRAM(s, tmp.data(), tmp.size());
            shadow_owner_[s] = b;
            idx += 1;
        }
    }
}

// ===== drained-state maintenance & utilities =====

void Builder::RecomputeBatching() {
    batches_needed_        = std::max(1, std::min(bmap_.NumBatches(), kMaxBatches));
    required_active_fifos_ = batches_needed_;  // kept for introspection
}

bool Builder::BatchTotallyDrained(int b) const {
    // 1) No pending DRAM slices for ANY spine of batch b
    for (int s = 0; s < kNumSpines; ++s) {
        const auto& addrs = bmap_.Get(b, s);
        if (next_addr_idx_[b][s] < static_cast<int>(addrs.size())) {
            return false;  // still more slices available in DRAM
        }
    }
    // 2) No SHADOW bank currently holds batch b
    for (int s = 0; s < kNumSpines; ++s) {
        if (shadow_owner_[s] == b) return false;
    }
    // 3) If batch b is currently ACTIVE, its ACTIVE banks must be empty
    if (active_batch_ == b) {
        for (int s = 0; s < kNumSpines; ++s) {
            if (!in_buf_.Empty(s)) return false;  // ACTIVE still has data
        }
    }
    // 4) Its FIFO is empty
    if (!fifos_[b].empty()) return false;

    // If all above hold, nothing else can generate new entries for batch b.
    return true;
}

void Builder::UpdateDrainStatus() {
    // Recompute 'totally_drained_' in a monotonic way and adjust pointers.
    for (int b = 0; b < batches_needed_; ++b) {
        if (totally_drained_[b]) continue;
        if (BatchTotallyDrained(b)) {
            totally_drained_[b] = true;

            // If the drained batch was ACTIVE, clear it.
            if (active_batch_ == b) active_batch_ = -1;

            // If the drained batch is being prepared, advance prep pointer
            if (prep_batch_ == b) {
                for (int step = 0; step < batches_needed_; ++step) {
                    int cand = (prep_batch_ + 1 + step) % batches_needed_;
                    if (!totally_drained_[cand]) { prep_batch_ = cand; break; }
                }
            }
        }
    }

    // If there is no ACTIVE batch and some non-drained batch already has SHADOW ready,
    // you can optionally fast-path an immediate swap next step via Stage5.
    // (We keep Stage5 as the single place performing swaps.)
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
        << ", drained=[";
    for (int b = 0; b < batches_needed_; ++b) {
        oss << (totally_drained_[b] ? '1' : '0');
        if (b + 1 < batches_needed_) oss << ",";
    }
    oss << "]}";
    return oss.str();
}

} // namespace sf
