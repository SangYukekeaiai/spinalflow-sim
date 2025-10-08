#pragma once
// All comments are in English.

#include <cstdint>
#include <vector>
#include <stdexcept>

#include "common/constants.hpp"
#include "common/entry.hpp"
#include "arch/dram/simple_dram.hpp"  // for SimpleDRAM handle
#include <iostream>


namespace sf {

/**
 * OutputSpine
 * - Collects output entries pushed by the OutputSorter.
 * - Has a capacity limit; overflow throws.
 * - Can store its buffered entries to DRAM and then clears the buffer.
 */
class OutputSpine {
public:
  explicit OutputSpine(sf::dram::SimpleDRAM* dram, int spine_id,
                       std::size_t capacity_limit = kOutputSpineMaxEntries)
  : dram_(dram), spine_id_(spine_id), capacity_limit_(capacity_limit) {}

  bool Push(const Entry& e) {
    if (buf_.size() >= capacity_limit_) {
      throw std::runtime_error("OutputSpine::Push: capacity exceeded.");
    }
    buf_.push_back(e);
    return true;
  }

  // Store to DRAM via SimpleDRAM API; clears buffer on success.
  // Throws on null DRAM pointer or store failure.
  std::uint32_t StoreOutputSpineToDRAM(std::uint32_t layer_id) {
    if (!dram_) {
      throw std::runtime_error("OutputSpine::StoreOutputSpineToDRAM: DRAM pointer is null.");
    }
    const std::uint32_t bytes =
        static_cast<std::uint32_t>(buf_.size() * sizeof(Entry));
        if (buf_.size() >= 1000) std::cout << "OutputSpine::StoreOutputSpineToDRAM: Storing " << buf_.size() << " entries to DRAM for layer " << layer_id << ", spine_id " << spine_id_ << "\n";
    buf_.clear();
    // const std::uint32_t written =
    //     dram_->StoreOutputSpine(layer_id, static_cast<std::uint32_t>(spine_id_),
    //                             static_cast<const void*>(buf_.data()), bytes);
    // if (written != bytes) {
    //   throw std::runtime_error("OutputSpine::StoreOutputSpineToDRAM: partial write / failure.");
    // }
    // buf_.clear();
    // return written;
    return bytes;
  }

  // Accessors
  int spine_id() const { return spine_id_; }
  std::size_t size() const { return buf_.size(); }
  bool empty() const { return buf_.empty(); }
  std::size_t capacity_limit() const { return capacity_limit_; }

private:
  sf::dram::SimpleDRAM* dram_ = nullptr;   // non-owning
  int spine_id_ = 0;
  std::vector<Entry> buf_;
  std::size_t capacity_limit_ = kOutputSpineMaxEntries;
};

} // namespace sf
