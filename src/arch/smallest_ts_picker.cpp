#include "arch/smallest_ts_picker.hpp"

#include <algorithm>

namespace sf {

void SmallestTsPicker::Clear() {
    entries_.clear();
}

void SmallestTsPicker::Push(const Entry& e) {
    entries_.push_back(e);
}

bool SmallestTsPicker::PopSmallest(Entry& out) {
    if (entries_.empty()) {
        return false;
    }

    auto it = std::min_element(entries_.begin(), entries_.end(), [](const Entry& a, const Entry& b) {
        if (a.ts != b.ts) {
            return a.ts < b.ts;
        }
        return a.neuron_id < b.neuron_id;
    });

    if (it == entries_.end()) {
        return false;
    }

    out = *it;
    entries_.erase(it);
    return true;
}

} // namespace sf

