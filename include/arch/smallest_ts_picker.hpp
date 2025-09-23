#pragma once

#include <vector>

#include "common/entry.hpp"

namespace sf {

/**
 * SmallestTsPicker
 *
 * Collects PE spike outputs for a cycle and lets the caller dequeue the
 * smallest timestamp first. The picker keeps ties stable by neuron id so the
 * behaviour is deterministic across runs.
 */
class SmallestTsPicker {
public:
    SmallestTsPicker() = default;

    // Clear all buffered entries.
    void Clear();

    // Append an entry to the pool; always succeeds.
    void Push(const Entry& e);

    // True when no entries are buffered.
    bool Empty() const { return entries_.empty(); }

    // Pop the entry with the smallest timestamp (tie-breaking on neuron id).
    // Returns false if the pool is empty.
    bool PopSmallest(Entry& out);

private:
    std::vector<Entry> entries_;
};

} // namespace sf

