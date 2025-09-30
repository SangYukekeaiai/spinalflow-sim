#include "arch/dram/weight_loader.hpp"

namespace sf { namespace dram {

bool WeightLoader::LoadOCGroupTo(FilterBuffer& fb, std::uint16_t oc_group) {
  if (!fmt_) return false;

  // Compute how many rows we expect for this oc_group.
  const std::size_t rows_expected =
      static_cast<std::size_t>(shape_.IC) * shape_.KH * shape_.KW;

  if (rows_expected == 0) {
    throw std::invalid_argument("WeightLoader: rows_expected is zero (invalid shape)");
  }

  // Each row is exactly 128 bytes (FilterBuffer::kRowBytes).
  const std::size_t bytes_total = rows_expected * FilterBuffer::kRowBytes;

  // Staging buffer: one contiguous blob containing all rows for this oc_group.
  std::vector<std::uint8_t> staging(bytes_total, 0);

  // Open the weights range for this layer, matching by oc_group only (MVP).
  WeightMatchPolicy pol{};
  pol.oc_group  = oc_group;  // required
  pol.check_inC = false;     // optional filters disabled for MVP
  pol.check_hw  = false;

  if (!reader_.open_weight(layer_id_, pol)) {
    // No weights range (or empty) for this layer.
    return false;
  }

  // Scan segments and place each payload row into the correct slot in staging.
  std::vector<std::uint8_t> line;
  SegmentHeader hdr{};
  bool any = false;

  while (reader_.read_next(line, &hdr)) {
    // Decode (ky, kx) from flattened kk-index stored in logical_spine_id.
    const std::uint16_t kk_idx = hdr.logical_spine_id;
    const std::uint16_t ky     = static_cast<std::uint16_t>(kk_idx / shape_.KW);
    const std::uint16_t kx     = static_cast<std::uint16_t>(kk_idx % shape_.KW);
    const std::uint16_t inC    = hdr.aux0; // packer must set aux0=inC

    check_indices(inC, ky, kx);

    // Payload must be exactly one row of 128 int8 weights.
    const std::size_t hdrB = fmt_->header_bytes();
    const std::size_t plB  = fmt_->payload_bytes(hdr);
    if (plB != FilterBuffer::kRowBytes) {
      throw std::invalid_argument("WeightLoader: payload size is not 128 bytes for a weight row");
    }

    // Compute destination offset in staging.
    const std::size_t lr  = local_row(inC, ky, kx);
    const std::size_t off = lr * FilterBuffer::kRowBytes;

    // Copy row bytes from the line's entries area.
    const std::uint8_t* src = line.data() + hdrB;
    std::memcpy(staging.data() + off, src, FilterBuffer::kRowBytes);
    any = true;
  }

  if (!any) {
    // No segments found for this oc_group; nothing to load.
    return false;
  }

  // Push the whole blob into FilterBuffer. It will map rows [0..rows_expected)
  // into banks/rows internally (interleaved).
  fb.LoadFromDRAM(reinterpret_cast<const int8_t*>(staging.data()), staging.size());
  return true;
}

}} // namespace sf::dram
