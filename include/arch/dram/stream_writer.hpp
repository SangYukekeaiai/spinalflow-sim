#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <stdexcept>
#include <cstring>

#include "arch/dram/dram_common.hpp"
#include "arch/dram/dram_format.hpp"

namespace sf { namespace dram {

// StreamWriter appends a fully formed segment (header + payload) to a byte buffer.
// For FixedStride formats, it also pads up to the fixed line size.
// All comments are in English.
class StreamWriter {
public:
  StreamWriter(const DramFormat& fmt, std::vector<std::uint8_t>& image)
  : fmt_(fmt), img_(image) {}

  void append(const SegmentHeader& hdr, const std::uint8_t* payload, std::size_t payload_len) {
    if (!payload && payload_len) throw std::invalid_argument("StreamWriter::append: null payload");
    if (fmt_.header_bytes() != sizeof(SegmentHeader))
      throw std::invalid_argument("StreamWriter::append: unexpected header size");
    // hdr.size should match payload_len / entry_bytes()
    const std::size_t expected = static_cast<std::size_t>(hdr.size) * fmt_.entry_bytes();
    if (expected != payload_len)
      throw std::invalid_argument("StreamWriter::append: payload length mismatch");

    // Compute total line length according to format (may include padding)
    const std::size_t line_bytes = fmt_.line_bytes(hdr);
    const std::size_t headerB    = fmt_.header_bytes();

    if (line_bytes < headerB + payload_len)
      throw std::invalid_argument("StreamWriter::append: line_bytes too small");

    const std::size_t pad = line_bytes - headerB - payload_len;
    const std::size_t old_size = img_.size();
    img_.resize(old_size + line_bytes);

    // Copy header
    std::memcpy(img_.data() + old_size, &hdr, headerB);
    // Copy payload
    if (payload_len) {
      std::memcpy(img_.data() + old_size + headerB, payload, payload_len);
    }
    // Pad if needed
    if (pad) {
      std::memset(img_.data() + old_size + headerB + payload_len, 0, pad);
    }
  }

private:
  const DramFormat&          fmt_;
  std::vector<std::uint8_t>& img_;
};

}} // namespace sf::dram
