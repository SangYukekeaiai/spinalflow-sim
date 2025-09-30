#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <cstring>
#include <optional>
#include <stdexcept>

#include "arch/dram/dram_common.hpp"
#include "arch/dram/dram_format.hpp"
#include "arch/dram/layer_directory.hpp"

namespace sf { namespace dram {

// Light-weight view over a contiguous DRAM image.
// All comments are in English.
class DramImage {
public:
  void bind(const std::uint8_t* base, std::size_t bytes) {
    base_ = base; size_ = bytes;
  }
  const std::uint8_t* data() const { return base_; }
  std::size_t         size() const { return size_; }
  bool                empty() const { return size_ == 0 || base_ == nullptr; }
private:
  const std::uint8_t* base_ = nullptr;
  std::size_t         size_ = 0;
};

// Optional weight matching policy.
// By default we only match on oc_group; inC/hw checks are disabled.
struct WeightMatchPolicy {
  std::uint16_t oc_group = 0; // required

  bool          check_inC = false;
  std::uint16_t inC       = 0;

  bool          check_hw  = false;         // (h,w) packed as (h<<8)|w
  std::uint16_t hw_packed = 0xFFFF;        // 0xFFFF can be used as don't-care
};

// StreamReader scans a region and returns segments that match the key/policy.
// MVP policy: linear scan within the layer's inputs or weights range.
class StreamReader {
public:
  StreamReader(const DramFormat& fmt,
               const DramImage&  img,
               const LayerDirectory& dir)
  : fmt_(fmt), img_(img), dir_(dir) {}

  // Open a spine stream: select the layer's inputs range and match by logical_spine_id.
  bool open_spine(std::uint16_t layer, std::uint16_t spine_id);

  // Open a weight stream: select the layer's weights range and match by policy (oc_group required).
  bool open_weight(std::uint16_t layer, const WeightMatchPolicy& pol);

  // Read next matching segment; copies the entire line into out_line.
  // If out_hdr!=nullptr, also returns the parsed header.
  bool read_next(std::vector<std::uint8_t>& out_line, SegmentHeader* out_hdr = nullptr);

  // Region bounds (for debugging/inspection).
  std::uint64_t region_begin() const { return region_begin_; }
  std::uint64_t region_end()   const { return region_end_;   }
  std::uint64_t cursor()       const { return cursor_;       } // current scan pointer

private:
  bool open_common(std::uint8_t kind, std::uint16_t layer,
                   std::uint64_t begin, std::uint64_t end,
                   std::uint16_t spine_or_hw,
                   std::optional<WeightMatchPolicy> wpol);

  bool match(const SegmentHeader& h) const;

private:
  const DramFormat&     fmt_;
  const DramImage&      img_;
  const LayerDirectory& dir_;

  // Key state
  std::uint8_t  kind_      = SEG_SPINE;    // SEG_SPINE or SEG_WEIGHT
  std::uint16_t layer_id_  = 0;
  std::uint16_t spine_id_  = 0;            // for spines; or hw if policy.check_hw
  std::optional<WeightMatchPolicy> wpol_;  // weight match options

  // Region scanning state
  std::uint64_t cursor_       = 0;         // absolute offset of next scan
  std::uint64_t region_begin_ = 0;
  std::uint64_t region_end_   = 0;

  bool opened_ = false;                    // true after successful open_*
};

}} // namespace sf::dram
