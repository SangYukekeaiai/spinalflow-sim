#include "arch/smallest_ts_picker.hpp"

// All comments are in English.
#include <algorithm>
#include "core/clock.hpp"          // to access Core API
#include "arch/output_queue.hpp"   // OutputQueue API

namespace sf {

bool SmallestTsPicker::Stage2Write(const Entry& e) {
    if (!core_) return false;
    // Only accept when inter-stage valid is true.
    if (!core_->st1_st2_valid()) return false;
    // Directly append to the pool.
    entries_.push_back(e);
    return true;
}

bool SmallestTsPicker::PopSmallest(Entry& out) {
    if (entries_.empty()) return false;

    auto it = std::min_element(entries_.begin(), entries_.end(),
        [](const Entry& a, const Entry& b) {
            if (a.ts != b.ts) return a.ts < b.ts;
            return a.neuron_id < b.neuron_id;
        });

    if (it == entries_.end()) return false;
    out = *it;
    entries_.erase(it);
    return true;
}

// All comments are in English.
bool SmallestTsPicker::run() {
    if (!core_) return false;

    // If upstream (S0) is invalid, stall and pull S1->S2 valid low.
    if (!core_->st0_st1_valid()) {
        if (core_->st1_st2_valid()) core_->SetSt1St2Valid(false);
        return false;
    }

    bool progressed = false;
    OutputQueue& outq = core_->output_queue();

    // If we have pending entries to drain to S0, block S2 to avoid overflow.
    if (!entries_.empty()) {
        if (core_->st1_st2_valid()) {
            core_->SetSt1St2Valid(false); // block Stage-2 during draining
            progressed = true;            // state changed
        }
    }

    // Drain up to per_cycle_budget_ entries in ascending timestamp order.
    std::size_t sent = 0;
    while (sent < per_cycle_budget_) {
        if (entries_.empty()) break;
        if (outq.full())      break;

        Entry e{};
        if (!PopSmallest(e)) break;

        if (!outq.push_entry(e)) {
            // Downstream capacity changed mid-cycle; put it back and stop.
            entries_.push_back(e);
            break;
        }
        ++sent;
        progressed = true;
    }

    // Open the upstream gate (allow S2) only when:
    // (1) local pool is empty, and (2) S0 still has space.
    const bool can_open_upstream = entries_.empty() && !outq.full();
    if (can_open_upstream) {
        if (!core_->st1_st2_valid()) {
            core_->SetSt1St2Valid(true);  // allow Stage-2 to produce again
            progressed = true;            // state changed
        }
    } else {
        if (core_->st1_st2_valid()) {
            core_->SetSt1St2Valid(false); // keep Stage-2 blocked
            progressed = true;            // state changed
        }
    }

    return progressed;
}

} // namespace sf
