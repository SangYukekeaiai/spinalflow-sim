#include "arch/output_queue.hpp"

namespace sf {

OutputQueue::OutputQueue(std::size_t capacity)
  : buf_(capacity), head_(0), tail_(0), size_(0) {}

bool OutputQueue::push_entry(const Entry& e) {
    if (full()) return false;
    buf_[tail_] = e;
    tail_ = (tail_ + 1) % buf_.size();
    ++size_;
    return true;
}

bool OutputQueue::pop(Entry& out) {
    if (empty()) return false;
    out = buf_[head_];
    head_ = (head_ + 1) % buf_.size();
    --size_;
    return true;
}

bool OutputQueue::front(Entry& out) const {
    if (empty()) return false;
    out = buf_[head_];
    return true;
}

void OutputQueue::clear() {
    head_ = tail_ = size_ = 0;
}

} // namespace sf
