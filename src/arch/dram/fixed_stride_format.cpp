#include <cstring>
#include <stdexcept>
#include "arch/dram/fixed_stride_format.hpp"

namespace sf { namespace dram {

FixedStrideFormat::FixedStrideFormat(std::size_t headerB,
                                     std::size_t entryB,
                                     std::size_t maxEntries)
  : headerB_(headerB), entryB_(entryB), maxEntries_(maxEntries) {
  // Basic sanity checks
  if (headerB_ == 0 || entryB_ == 0 || maxEntries_ == 0) {
    throw std::invalid_argument("FixedStrideFormat: zero size not allowed");
  }
}

void FixedStrideFormat::parse_header(const std::uint8_t* hdr_raw,
                                     SegmentHeader&      out) const {
  if (!hdr_raw) throw std::invalid_argument("FixedStrideFormat::parse_header: null hdr_raw");
  if (headerB_ != sizeof(SegmentHeader)) {
    // We require binary-compatibility for simplicity here.
    throw std::invalid_argument("FixedStrideFormat::parse_header: unexpected header_bytes");
  }
  std::memcpy(&out, hdr_raw, sizeof(SegmentHeader));
  if (out.size > maxEntries_) {
    throw std::invalid_argument("FixedStrideFormat::parse_header: size > maxEntries");
  }
}

const std::uint8_t* FixedStrideFormat::entries_ptr(const std::uint8_t* line_base) const {
  if (!line_base) throw std::invalid_argument("FixedStrideFormat::entries_ptr: null line_base");
  return line_base + headerB_;
}

std::size_t FixedStrideFormat::payload_bytes(const SegmentHeader& hdr) const {
  return static_cast<std::size_t>(hdr.size) * entryB_;
}

std::size_t FixedStrideFormat::line_bytes(const SegmentHeader& /*hdr*/) const {
  // Fixed stride ignores hdr.size for hop distance.
  return headerB_ + maxEntries_ * entryB_;
}

}} // namespace sf::dram
