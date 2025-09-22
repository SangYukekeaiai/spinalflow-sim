#include "arch/intermediate_fifo.hpp"

namespace sf {

bool IntermediateFIFO::push(const Entry& e) {
  if (full()) return false;
  buf_[(head_ + size_) % kCapacityEntries] = e;
  ++size_;
  return true;
}

std::optional<Entry> IntermediateFIFO::front() const {
  if (empty()) return std::nullopt;
  return buf_[head_];
}

bool IntermediateFIFO::pop() {
  if (empty()) return false;
  head_ = (head_ + 1) % kCapacityEntries;
  --size_;
  return true;
}

void IntermediateFIFO::clear() {
  head_ = 0;
  size_ = 0;
}

} // namespace sf
