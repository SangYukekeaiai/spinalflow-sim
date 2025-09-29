#pragma once
#include <cstddef>
#include <cstdint>
#include "arch/dram/dram_common.hpp"

namespace sf { namespace dram {

/**
 * DramFormat
 * Abstract interface for DRAM line/segment layout.
 *
 * A "line" == one segment: [Header][Entries...]
 * Implementations differ in how "line_bytes" is computed:
 *  - FixedStride: line_bytes = header_bytes + max_entries_per_line * entry_bytes
 *  - Packed:      line_bytes = header_bytes + hdr.size * entry_bytes
 */
class DramFormat {
public:
  virtual ~DramFormat() = default;

  // Size of header in bytes (usually sizeof(SegmentHeader))
  virtual std::size_t header_bytes() const = 0;

  // Size of one Entry in bytes (usually sizeof(sf::Entry))
  virtual std::size_t entry_bytes() const = 0;

  // Maximum entries per line (e.g., 128). For Packed format this is still a limit.
  virtual std::size_t max_entries_per_line() const = 0;

  // Parse the header from raw bytes into a SegmentHeader struct.
  // Must read exactly header_bytes() from hdr_raw.
  virtual void parse_header(const std::uint8_t* hdr_raw,
                            SegmentHeader&      out) const = 0;

  // Return pointer to the start of entries array within a line buffer (line_base).
  // For both formats, entries start immediately after header.
  virtual const std::uint8_t* entries_ptr(const std::uint8_t* line_base) const = 0;

  // Number of entry bytes that are valid for this line (based on header).
  // For Packed:  hdr.size * entry_bytes()
  // For Stride:  hdr.size * entry_bytes()   (padding may exist beyond that)
  virtual std::size_t payload_bytes(const SegmentHeader& hdr) const = 0;

  // Total line bytes including any padding to hop to the next line.
  // For Packed:  header_bytes + hdr.size * entry_bytes
  // For Stride:  header_bytes + max_entries_per_line * entry_bytes
  virtual std::size_t line_bytes(const SegmentHeader& hdr) const = 0;
};

}} // namespace sf::dram
