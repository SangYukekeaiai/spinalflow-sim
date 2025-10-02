#include "arch/dram/stream_reader.hpp"
#include <algorithm>

namespace sf { namespace dram {

static inline std::uint64_t hop_next(const DramFormat& fmt,
                                     std::uint64_t pos,
                                     std::uint64_t end,
                                     const SegmentHeader& hdr)
{
  const std::size_t hop = fmt.line_bytes(hdr);
  const std::uint64_t next = pos + static_cast<std::uint64_t>(hop);
  return (next > end) ? end : next;
}

bool StreamReader::open_input(std::uint16_t layer, std::uint16_t spine_id) {
  const auto& r = dir_.input_range(static_cast<int>(layer));
  if (r.empty() || img_.empty()) return false;
  return open_common(SEG_INPUT, layer, r.begin, r.end, /*spine_or_hw*/spine_id, std::nullopt);
}

bool StreamReader::open_output(std::uint16_t layer, std::uint16_t spine_id) {
  const auto& r = dir_.output_range(static_cast<int>(layer));  // requires LayerDirectory change
  if (r.empty() || img_.empty()) return false;
  return open_common(SEG_OUTPUT, layer, r.begin, r.end, /*spine_or_hw*/spine_id, std::nullopt);
}


bool StreamReader::open_weight(std::uint16_t layer, const WeightMatchPolicy& pol) {
  const auto& r = dir_.weights_range(static_cast<int>(layer));
  if (r.empty() || img_.empty()) return false;
  return open_common(SEG_WEIGHT, layer, r.begin, r.end, /*spine_or_hw*/pol.hw_packed, pol);
}

bool StreamReader::open_common(std::uint8_t kind, std::uint16_t layer,
                               std::uint64_t begin, std::uint64_t end,
                               std::uint16_t spine_or_hw,
                               std::optional<WeightMatchPolicy> wpol)
{
  if (begin >= end || end > img_.size()) return false;

  kind_      = kind;
  layer_id_  = layer;
  spine_id_  = spine_or_hw;
  wpol_      = std::move(wpol);

  region_begin_ = begin;
  region_end_   = end;
  cursor_       = region_begin_;
  opened_       = true;
  return true;
}

bool StreamReader::match(const SegmentHeader& h) const {
  if (h.kind != kind_) return false;
  if (h.layer_id != layer_id_) return false;

  if (kind_ == SEG_INPUT || kind_ == SEG_OUTPUT) {
    // Spine: matching only by logical_spine_id inside the layer's input range.
    return (h.logical_spine_id == spine_id_);
  }

  // SEG_WEIGHT
  if (!wpol_.has_value()) return false; // safety
  const auto& p = *wpol_;

  // Required: oc_group via aux1
  if (h.aux1 != p.oc_group) return false;

  // Optional: inC via aux0
  if (p.check_inC && h.aux0 != p.inC) return false;

  // Optional: (h,w) packed as logical_spine_id
  if (p.check_hw && h.logical_spine_id != p.hw_packed) return false;

  return true;
}

bool StreamReader::read_next(std::vector<std::uint8_t>& out_line, SegmentHeader* out_hdr) {
  if (!opened_) return false;
  if (cursor_ >= region_end_) return false;

  const std::uint8_t* const base = img_.data();

  std::uint64_t pos = cursor_;
  while (pos + fmt_.header_bytes() <= region_end_) {
    const std::uint8_t* line = base + pos;

    SegmentHeader hdr{};
    fmt_.parse_header(line, hdr);

    const std::size_t lineB = fmt_.line_bytes(hdr);
    if (pos + lineB > region_end_) { // truncated line at region end
      cursor_ = region_end_;
      return false;
    }

    if (match(hdr)) {
      out_line.resize(lineB);
      std::memcpy(out_line.data(), line, lineB);
      if (out_hdr) *out_hdr = hdr;
      cursor_ = pos + lineB; // advance to next line
      return true;
    }

    pos += lineB; // skip non-matching line
  }

  cursor_ = region_end_;
  return false;
}

}} // namespace sf::dram
