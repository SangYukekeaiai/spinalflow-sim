#pragma once
#include <vector>
#include <cstddef>
#include "common/constants.hpp"
#include "common/entry.hpp"
#include "core/core_iface.hpp"

namespace sf {

class OutputQueue {
public:
    explicit OutputQueue(std::size_t capacity = kDefaultOutputQueueCapacity);

    // Producer side: push an entry (false if full).
    bool push_entry(const Entry& e);

    // Stage0 replacement: try to drain one entry to the sink via CoreIface.
    // Returns true if an entry was successfully delivered and popped.
    bool run();

    // Register the core so we can reach the output sink without exposing details.
    void RegisterCore(CoreIface* core) { core_ = core; }

    // Introspection
    bool   empty() const { return size_ == 0; }
    bool   full()  const { return size_ == buf_.size(); }
    size_t size()  const { return size_; }
    size_t capacity() const { return buf_.size(); }

    // Clear the entire queue (use with care).
    void   clear();

private:
    // Consumer utilities are private after Stage B hardening.
    bool pop(Entry& out);            // false if empty
    bool front(Entry& out) const;    // peek at head (false if empty)

private:
    std::vector<Entry> buf_;
    std::size_t head_ = 0;
    std::size_t tail_ = 0;
    std::size_t size_ = 0;

    CoreIface* core_ = nullptr;
};

} // namespace sf
