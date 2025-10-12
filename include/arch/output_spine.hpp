// arch/output_spine.hpp
// All comments are in English.
#pragma once
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include "common/entry.hpp"
#include "arch/dram/simple_dram.hpp"

namespace sf {

class OutputSpine {
public:

  explicit OutputSpine(sf::dram::SimpleDRAM* dram,
                       std::size_t capacity_limit = kOutputSpineMaxEntries)
    : dram_(dram), capacity_limit_(capacity_limit) {}


  
  void SetSpineID(int spine_id) { spine_id_ = spine_id; }
  bool Push(const Entry& e) {
    if (buf_.size() >= capacity_limit_) {
      throw std::runtime_error("OutputSpine::Push: capacity exceeded.");
    }
    buf_.push_back(e);
    return true;
  }

  std::size_t size() const { return buf_.size(); }

  // NEW: optional out_cycles to return timing based on wire bytes.
  std::uint32_t StoreOutputSpineToDRAM(std::uint32_t layer_id) {
    if (!dram_) {
      throw std::runtime_error("OutputSpine::StoreOutputSpineToDRAM: DRAM pointer is null.");
    }

    // Calculate wire bytes and cycles BEFORE clearing.
    const std::size_t entries = buf_.size();



    // Return "bytes" for compatibility (previous behavior used sizeof(Entry)).
    // If you want to keep that semantic, do so; timing still uses wire_bytes.
    const std::uint32_t bytes = static_cast<std::uint32_t>(entries * sizeof(Entry));
    buf_.clear();

    return bytes;
  }

private:
  static constexpr std::size_t kOutputSpineMaxEntries = 1u << 20; // example
  static inline uint64_t CeilDivU64(uint64_t a, uint64_t b) { return (a + b - 1) / b; }

  sf::dram::SimpleDRAM* dram_ = nullptr;
  int spine_id_ = 0;
  std::size_t capacity_limit_ = kOutputSpineMaxEntries;
  std::vector<Entry> buf_;
};

} // namespace sf
