#pragma once
// All comments are in English.

#include <cstddef>
#include <optional>
#include "common/constants.hpp"
#include "common/entry.hpp"

namespace sf {

// Compute FIFO capacity in entries at namespace scope so it is an ICE everywhere.
inline constexpr std::size_t kInterFifoCapacityEntries =
    (sf::kInterFifoCapacityBytes / sizeof(Entry));

/**
 * IntermediateFIFO
 *
 * A simple circular FIFO with fixed byte capacity configured via common/constants.hpp:
 *   - kInterFifoCapacityBytes
 *
 * Capacity in entries is derived as: kInterFifoCapacityBytes / sizeof(Entry).
 */
class IntermediateFIFO {
public:
  IntermediateFIFO() = default;

  // Push one entry; returns false if the FIFO is full.
  bool push(const Entry& e);

  // Read-only peek; returns std::nullopt if empty.
  std::optional<Entry> front() const;

  // Pop the head entry; returns false if empty.
  bool pop();

  // Status helpers.
  bool empty() const { return size_ == 0; }
  bool full()  const { return size_ == kInterFifoCapacityEntries; }
  std::size_t size() const { return size_; }

  // Clear all entries (no destructor side-effects).
  void clear();

private:
  // Use the namespace-level constant directly as the array bound to avoid
  // “member constant not usable as constant” issues on some compilers.
  Entry        buf_[kInterFifoCapacityEntries]{};
  std::size_t  head_ = 0;  // index of the oldest entry
  std::size_t  size_ = 0;  // number of valid entries
};

} // namespace sf
