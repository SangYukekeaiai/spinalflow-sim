#pragma once
#include <cstddef>
#include <cstdint>
#include <optional>
#include "common/entry.hpp"

namespace sf {

/**
 * IntermediateFIFO
 *
 * A simple circular FIFO buffer with fixed byte capacity.
 * Each Entry stores an 8-bit timestamp and a 32-bit neuron id.
 */
class IntermediateFIFO {
public:
  static constexpr std::size_t kCapacityBytes   = 256;
  static constexpr std::size_t kEntrySize       = sizeof(Entry);
  static constexpr std::size_t kCapacityEntries = kCapacityBytes / kEntrySize;

  IntermediateFIFO() = default;

  // Push one entry; return false if full.
  bool push(const Entry& e);

  // Read-only peek; nullopt if empty.
  std::optional<Entry> front() const;

  // Pop the head entry; return false if empty.
  bool pop();

  bool empty() const { return size_ == 0; }
  bool full()  const { return size_ == kCapacityEntries; }
  std::size_t size() const { return size_; }

  // Clear all entries.
  void clear();

private:
  Entry buf_[kCapacityEntries];
  std::size_t head_ = 0;  // index of oldest entry
  std::size_t size_ = 0;  // number of valid entries
};

} // namespace sf
