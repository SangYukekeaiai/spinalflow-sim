#pragma once
#include "arch/dram/dram_format.hpp"

namespace sf { namespace dram {

/**
 * FixedStrideFormat (Plan A)
 * Next line = curr + (header_bytes + max_entries * entry_bytes).
 * Entries beyond hdr.size within the line are padding and must be ignored.
 */
class FixedStrideFormat final : public DramFormat {
public:
  FixedStrideFormat(std::size_t headerB,
                    std::size_t entryB,
                    std::size_t maxEntries);

  std::size_t header_bytes() const override { return headerB_; }
  std::size_t entry_bytes()  const override { return entryB_;  }
  std::size_t max_entries_per_line() const override { return maxEntries_; }

  void parse_header(const std::uint8_t* hdr_raw,
                    SegmentHeader&      out) const override;

  const std::uint8_t* entries_ptr(const std::uint8_t* line_base) const override;

  std::size_t payload_bytes(const SegmentHeader& hdr) const override;
  std::size_t line_bytes(const SegmentHeader& hdr) const override;

private:
  std::size_t headerB_;
  std::size_t entryB_;
  std::size_t maxEntries_;
};

}} // namespace sf::dram
