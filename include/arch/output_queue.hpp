#pragma once
#include <vector>
#include <cstddef>
#include "common/output_token.hpp"

namespace sf {

// A bounded ring buffer of OutputToken with simple push/pop semantics.
// This is single-producer/single-consumer friendly; add guards if multi-threaded.
class OutputQueue {
public:
    explicit OutputQueue(std::size_t capacity);

    bool push(const OutputToken& x); // returns false if full
    bool pop(OutputToken& out);      // returns false if empty

    bool   empty() const { return size_ == 0; }
    bool   full()  const { return size_ == buf_.size(); }
    size_t size()  const { return size_; }
    size_t capacity() const { return buf_.size(); }

    // Optional: peek front without popping.
    bool   front(OutputToken& out) const;

    // Optional: clear queue (dangerous if someone is consuming).
    void   clear();

private:
    std::vector<OutputToken> buf_;
    std::size_t head_ = 0;
    std::size_t tail_ = 0;
    std::size_t size_ = 0;
};

} // namespace sf
