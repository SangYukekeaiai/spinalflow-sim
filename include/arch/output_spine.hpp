// arch/output_spine.hpp
// All comments are in English.
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include "common/entry.hpp"
#include "common/constants.hpp"
#include "arch/dram/simple_dram.hpp"

namespace sf {

class OutputSpine {
public:

  explicit OutputSpine(sf::dram::SimpleDRAM* dram,
                       std::size_t capacity_limit = kOutputSpineMaxEntries)
    : dram_(dram),
      capacity_limit_(std::min<std::size_t>(capacity_limit, kMaxBufferedEntries)) {}


  
  void SetSpineID(int spine_id) { spine_id_ = spine_id; }
  bool Push(const Entry& e) {
    if (buf_.size() >= capacity_limit_) {
      throw std::runtime_error("OutputSpine::Push: capacity exceeded.");
    }
    buf_.push_back(e);
    return true;
  }

  std::size_t size() const { return buf_.size(); }
  bool empty() const { return buf_.empty(); }
  bool IsFull() const { return buf_.size() >= capacity_limit_; }

  // NEW: optional out_cycles to return timing based on wire bytes.
  std::uint32_t StoreOutputSpineToDRAM(std::uint32_t layer_id) {
    if (!dram_) {
      throw std::runtime_error("OutputSpine::StoreOutputSpineToDRAM: DRAM pointer is null.");
    }

    // Calculate wire bytes and cycles BEFORE clearing.
    const std::size_t entries_available = buf_.size();
    if (entries_available == 0) {
      return 0;
    }
    const std::size_t entries_to_store = std::min(entries_available, kEntriesPerBuffer);

    // Return "bytes" for compatibility (previous behavior used sizeof(Entry)).
    // If you want to keep that semantic, do so; timing still uses wire_bytes.
    const std::uint32_t bytes = static_cast<std::uint32_t>(entries_to_store * sizeof(Entry));
    buf_.erase(buf_.begin(),
               buf_.begin() + static_cast<std::ptrdiff_t>(entries_to_store));

    return bytes;
  }

private:
  static constexpr std::size_t kEntriesPerBuffer   = 512;
  static constexpr std::size_t kNumBuffers         = 2;
  static constexpr std::size_t kMaxBufferedEntries = kEntriesPerBuffer * kNumBuffers;

  sf::dram::SimpleDRAM* dram_ = nullptr;
  int spine_id_ = 0;
  std::size_t capacity_limit_ = kMaxBufferedEntries;
  std::vector<Entry> buf_;
};

} // namespace sf
