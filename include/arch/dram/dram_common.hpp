#pragma once
#include <cstdint>

namespace sf { namespace dram {

// Region/kind of segment in DRAM layout.
// NOTE: SEG_SPINE is kept as a backward alias to SEG_INPUT to minimize code churn.
enum : std::uint8_t {
  SEG_INPUT  = 0,   // input spines region
  SEG_WEIGHT = 1,   // weight segments region
  SEG_OUTPUT = 2,   // output spines region

  // Backward-compatibility alias (legacy code may still reference SEG_SPINE).
  SEG_SPINE  = SEG_INPUT
};

// Unified on-disk header (MVP).
// All comments are in English.
struct SegmentHeader {
  std::uint8_t  version;          // header layout version; set to 2 for the new 3-way layout
  std::uint8_t  kind;             // SEG_INPUT, SEG_WEIGHT, or SEG_OUTPUT
  std::uint16_t layer_id;         // layer index (0-based)

  // For SEG_INPUT/SEG_OUTPUT:
  //   logical_spine_id = h*W + w (your definition)
  // For SEG_WEIGHT:
  //   logical_spine_id may encode (h<<8)|w for compactness if H,W<=255
  std::uint16_t logical_spine_id;

  // Number of valid entries in this segment (<= max_entries_per_line)
  std::uint8_t  size;

  // Segment counters for this logical stream
  std::uint8_t  seg_id;           // 0..seg_count-1
  std::uint8_t  seg_count;        // total number of segments for this stream

  // End-of-logical-stream flag (redundant with seg_id==seg_count-1, but handy)
  std::uint8_t  eol;              // 1 if this is the last segment

  // For SEG_WEIGHT we carry the identity (inC, oc_group).
  // For SEG_INPUT/SEG_OUTPUT these are reserved and must be 0.
  std::uint16_t aux0;             // SEG_WEIGHT: inC; SEG_INPUT/SEG_OUTPUT: reserved (0)
  std::uint16_t aux1;             // SEG_WEIGHT: oc_group (OC/128); SEG_INPUT/SEG_OUTPUT: reserved (0)

  // Keep 4-byte alignment for future fields (e.g., format_id, bitwidth, crc, etc.)
  std::uint32_t reserved;         // must be written as 0 for MVP
};

// Remove batch_index from StreamKey
struct StreamKey {
  std::uint8_t  kind;             // SEG_INPUT, SEG_WEIGHT, SEG_OUTPUT
  std::uint16_t layer_id;
  std::uint16_t spine_id_or_hw;   // input/output: h*W+w; weight: (h<<8)|w optional
  std::uint16_t inC;              // SEG_WEIGHT only
  std::uint16_t oc_group;         // SEG_WEIGHT only (OC/128)
};


}} // namespace sf::dram
