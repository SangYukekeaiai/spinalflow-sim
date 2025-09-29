#pragma once
// All comments are in English.

#include <vector>
#include <cstddef>                 // for std::size_t
#include "common/constants.hpp"
#include "common/entry.hpp"

namespace sf {

// Forward declarations to avoid heavy includes in the header.
class ClockCore;
class OutputQueue;

/**
 * SmallestTsPicker (Stage-1)
 *
 * Simplified modeling:
 *  - Stage-2 writes directly into a single pool 'entries_' when st1_st2_valid()==true.
 *  - On run():
 *      * If entries_ is non-empty and inter-stage valid is still true,
 *        switch valid to false (enter consuming phase).
 *      * Forward up to 'per_cycle_budget_' entries in ascending (ts, neuron_id)
 *        order to the downstream OutputQueue.
 *      * If entries_ becomes empty, set inter-stage valid to true.
 *
 * Notes:
 *  - No per-PE inbox; we do not enforce "one per PE per cycle".
 */
class SmallestTsPicker {
public:
    SmallestTsPicker() = default;

    // Reset local pool.
    void Clear() { entries_.clear(); }

    // Stage-2 writes one entry when Core's st1_st2_valid()==true.
    // Returns true on success; false otherwise.
    bool Stage2Write(const Entry& e);

    // For tests/legacy paths: append directly.
    void Push(const Entry& e) { entries_.push_back(e); }

    // True when no entries are pending in the local pool.
    bool Empty() const { return entries_.empty(); }

    // Register Core to access output_queue() and control inter-stage valid.
    void RegisterCore(ClockCore* core) { core_ = core; }

    // Optional throughput limiter per run() call.
    void SetPerCycleBudget(std::size_t n) { per_cycle_budget_ = n; }

    // Stage-1 behavior; returns true if any progress (sent > 0 or state change).
    bool run();

private:
    // Pop the smallest entry by (ts, neuron_id) from the local pool.
    bool PopSmallest(Entry& out);

private:
    std::vector<Entry> entries_;        // unified pool from Stage-2
    ClockCore*         core_ = nullptr; // not owned
    std::size_t        per_cycle_budget_ = static_cast<std::size_t>(-1); // unlimited by default
};

} // namespace sf
