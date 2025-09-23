#pragma once
#include <vector>
#include <cstddef>
#include "common/constants.hpp"
#include "common/entry.hpp"

namespace sf {

/**
 * A bounded ring buffer for Entries produced by PEs.
 * This class acts as the "output queue" stage: Stage1 converts a PE's result
 * into an Entry and pushes it here; a later stage drains the queue to DRAM.
 */
class OutputQueue {
public:
    explicit OutputQueue(std::size_t capacity = kDefaultOutputQueueCapacity);

    // Producer side (Builder Stage1)
    bool push_entry(const Entry& e);   // false if full

    // Consumer side (Builder Stage0 drain-to-DRAM)
    bool pop(Entry& out);              // false if empty
    bool front(Entry& out) const;      // peek

    // Introspection
    bool   empty() const { return size_ == 0; }
    bool   full()  const { return size_ == buf_.size(); }
    size_t size()  const { return size_; }
    size_t capacity() const { return buf_.size(); }

    // Clear the entire queue (use with care)
    void   clear();

private:
    std::vector<Entry> buf_;
    std::size_t head_ = 0;
    std::size_t tail_ = 0;
    std::size_t size_ = 0;
};

} // namespace sf
