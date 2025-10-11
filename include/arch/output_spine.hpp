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
  struct Timing {
    uint32_t bw_bytes_per_cycle = 160; // e.g., 128-bit bus
    uint32_t fixed_latency = 0;       // per store transaction
    uint32_t wire_entry_bytes = 5;    // e.g., ts:uint8 + nid:uint32
  };

  explicit OutputSpine(sf::dram::SimpleDRAM* dram, int spine_id,
                       std::size_t capacity_limit = kOutputSpineMaxEntries)
    : dram_(dram), spine_id_(spine_id), capacity_limit_(capacity_limit) {}

  void SetTiming(const Timing& t) { timing_ = t; }

  bool Push(const Entry& e) {
    if (buf_.size() >= capacity_limit_) {
      throw std::runtime_error("OutputSpine::Push: capacity exceeded.");
    }
    buf_.push_back(e);
    return true;
  }

  std::size_t size() const { return buf_.size(); }

  // NEW: optional out_cycles to return timing based on wire bytes.
  std::uint32_t StoreOutputSpineToDRAM(std::uint32_t layer_id,
                                       uint64_t* out_cycles = nullptr) {
    if (!dram_) {
      throw std::runtime_error("OutputSpine::StoreOutputSpineToDRAM: DRAM pointer is null.");
    }
    if (out_cycles) *out_cycles = 0;

    // Calculate wire bytes and cycles BEFORE clearing.
    const std::size_t entries = buf_.size();

    // If you want to reflect actual writer format, use wire_entry_bytes (default 5).
    const uint64_t wire_bytes = static_cast<uint64_t>(entries) *
                                static_cast<uint64_t>(std::max(1u, timing_.wire_entry_bytes));
    const uint32_t bw = std::max(1u, timing_.bw_bytes_per_cycle);
    const uint64_t data_cycles  = (wire_bytes + bw - 1) / bw;
    const uint64_t total_cycles = data_cycles + static_cast<uint64_t>(timing_.fixed_latency);

    // Return "bytes" for compatibility (previous behavior used sizeof(Entry)).
    // If you want to keep that semantic, do so; timing still uses wire_bytes.
    const std::uint32_t bytes = static_cast<std::uint32_t>(entries * sizeof(Entry));

    // Fake store; real code commented out previously.
    // const std::uint32_t written = dram_->StoreOutputSpine(layer_id,
    //                            static_cast<std::uint32_t>(spine_id_),
    //                            static_cast<const void*>(buf_.data()), bytes);
    // if (written != bytes) { throw ... }

    // Clear after "store".
    buf_.clear();

    if (out_cycles) *out_cycles = total_cycles;
    return bytes;
  }

private:
  static constexpr std::size_t kOutputSpineMaxEntries = 1u << 20; // example
  static inline uint64_t CeilDivU64(uint64_t a, uint64_t b) { return (a + b - 1) / b; }

  sf::dram::SimpleDRAM* dram_ = nullptr;
  int spine_id_ = 0;
  std::size_t capacity_limit_ = kOutputSpineMaxEntries;

  Timing timing_;               // NEW
  std::vector<Entry> buf_;
};

} // namespace sf
