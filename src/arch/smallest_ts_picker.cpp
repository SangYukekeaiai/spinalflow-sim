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

bool SmallestTsPicker::run() {
    if (!core_) return false;

    bool progressed = false;

    // If we have pending entries but the inter-stage valid is still true,
    // flip it to false to stall Stage-2 while we are consuming.
    if (!entries_.empty() && core_->st1_st2_valid()) {
        core_->SetSt1St2Valid(false);
        progressed = true; // state changed
    }

    // Deliver from local pool to OutputQueue in ascending order.
    OutputQueue& outq = core_->output_queue();
    std::size_t sent = 0;

    while (sent < per_cycle_budget_) {
        if (entries_.empty()) break;
        if (outq.full())      break;

        Entry e{};
        if (!PopSmallest(e)) break;

        if (!outq.push_entry(e)) {
            // Queue got full just now; put the entry back and stop.
            entries_.push_back(e);
            break;
        }
        ++sent;
        progressed = true;
    }

    // If pool is empty after sending, re-open the gate for Stage-2.
    if (entries_.empty()) {
        if (!core_->st1_st2_valid()) {
            core_->SetSt1St2Valid(true);
            progressed = true; // state changed
        }
    } else {
        // Still have pending entries; keep Stage-2 stalled.
        if (core_->st1_st2_valid()) {
            core_->SetSt1St2Valid(false);
            progressed = true; // state changed
        }
    }

    return progressed;
}

} // namespace sf
