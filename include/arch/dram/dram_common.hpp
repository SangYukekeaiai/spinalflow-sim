#pragma once
#include <cstdint>

namespace sf { namespace dram {

/**
 * SegmentHeader
 * Shared header definition parsed from each DRAM line/segment.
 * Keep it binary-compatible with the producer (driver/DRAM writer).
 */
struct SegmentHeader {
  // All comments in English
  uint16_t batch_id;           // batch identity
  uint16_t logical_spine_id;   // spine id within the layer (e.g., 0..8 for 3x3)
  uint8_t  psb_hint;           // optional PSB slot hint (0..15) or 0xFF if don't-care
  uint8_t  size;               // # valid entries in this line (0..max_entries_per_line)
  uint8_t  seg_id;             // segment index (0..seg_count-1)
  uint8_t  seg_count;          // total # segments for this logical spine (1..N)
  uint8_t  eol;                // 1 if last segment for this logical spine
  uint8_t  reserved0;          // padding/alignment
  uint32_t reserved1;          // padding/alignment
};

}} // namespace sf::dram
