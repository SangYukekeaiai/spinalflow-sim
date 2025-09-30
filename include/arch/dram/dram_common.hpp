#pragma once
#include <cstdint>

namespace sf { namespace dram {

// Use a single kind for both input and output spines; plus weights.
enum : std::uint8_t {
  SEG_SPINE  = 0,  // both input and output spines
  SEG_WEIGHT = 1   // weight segments
};

// Unified on-disk header (MVP).
// All comments are in English.
struct SegmentHeader {
  std::uint8_t  version;          // header layout version; start with 1
  std::uint8_t  kind;             // SEG_SPINE or SEG_WEIGHT
  std::uint16_t layer_id;         // layer index (0-based)

  // For SEG_SPINE:
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

  // For SEG_WEIGHT we carry the identity (inC, oc_group)
  std::uint16_t aux0;             // SEG_WEIGHT: inC; SEG_SPINE: reserved (0)
  std::uint16_t aux1;             // SEG_WEIGHT: oc_group (OC/128); SEG_SPINE: reserved (0)

  // Keep 4-byte alignment for future fields (e.g., format_id, bitwidth, crc, etc.)
  std::uint32_t reserved;         // must be written as 0 for MVP
};

// Software-only key to locate a logical stream (not serialized in DRAM).
struct StreamKey {
  std::uint8_t  kind;             // SEG_SPINE or SEG_WEIGHT
  std::uint16_t layer_id;

  // Runtime-only selector to pick the batch subregion from LayerDirectory
  std::uint16_t batch_index;      // not compared against any header field

  // For SEG_SPINE: h*W + w
  // For SEG_WEIGHT: optionally (h<<8)|w if you also want to filter by (h,w)
  std::uint16_t spine_id_or_hw;

  // For SEG_WEIGHT only
  std::uint16_t inC;              // input channel index
  std::uint16_t oc_group;         // output-channel group (OC/128)
};

}} // namespace sf::dram
