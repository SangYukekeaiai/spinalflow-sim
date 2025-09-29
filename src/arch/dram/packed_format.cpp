#include <cstring>
#include <stdexcept>
#include "arch/dram/packed_format.hpp"

namespace sf { namespace dram {

PackedFormat::PackedFormat(std::size_t headerB,
                           std::size_t entryB,
                           std::size_t maxEntries)
  : headerB_(headerB), entryB_(entryB), maxEntries_(maxEntries) {
  if (headerB_ == 0 || entryB_ == 0 || maxEntries_ == 0) {
    throw std::invalid_argument("PackedFormat: zero size not allowed");
  }
}

void PackedFormat::parse_header(const std::uint8_t* hdr_raw,
                                SegmentHeader&      out) const {
  if (!hdr_raw) throw std::invalid_argument("PackedFormat::parse_header: null hdr_raw");
  if (headerB_ != sizeof(SegmentHeader)) {
    throw std::invalid_argument("PackedFormat::parse_header: unexpected header_bytes");
  }
  std::memcpy(&out, hdr_raw, sizeof(SegmentHeader));
  if (out.size > maxEntries_) {
    throw std::invalid_argument("PackedFormat::parse_header: size > maxEntries");
  }
}

const std::uint8_t* PackedFormat::entries_ptr(const std::uint8_t* line_base) const {
  if (!line_base) throw std::invalid_argument("PackedFormat::entries_ptr: null line_base");
  return line_base + headerB_;
}

std::size_t PackedFormat::payload_bytes(const SegmentHeader& hdr) const {
  return static_cast<std::size_t>(hdr.size) * entryB_;
}

std::size_t PackedFormat::line_bytes(const SegmentHeader& hdr) const {
  return headerB_ + static_cast<std::size_t>(hdr.size) * entryB_;
}

}} // namespace sf::dram
