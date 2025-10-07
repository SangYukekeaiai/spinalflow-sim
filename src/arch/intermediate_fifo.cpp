// All comments are in English.
#include "arch/intermediate_fifo.hpp"

namespace sf {

bool IntermediateFIFO::push(const Entry& e) {
  if (full()) return false;
  const std::size_t tail = (head_ + size_) % kInterFifoCapacityEntries;
  buf_[tail] = e;
  ++size_;
  return true;
}

std::optional<Entry> IntermediateFIFO::front() const {
  if (empty()) return std::nullopt;
  return buf_[head_];
}

bool IntermediateFIFO::pop() {
  if (empty()) return false;
  head_ = (head_ + 1) % kInterFifoCapacityEntries;
  --size_;
  return true;
}

void IntermediateFIFO::clear() {
  head_ = 0;
  size_ = 0;
}

} // namespace sf
