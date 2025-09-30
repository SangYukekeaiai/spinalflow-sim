#pragma once
#include <cstdint>
#include <vector>
#include <cstring>
#include <stdexcept>

#include "arch/dram/stream_reader.hpp"   // DramImage, StreamReader, LayerDirectory, SegmentHeader
#include "arch/dram/dram_format.hpp"     // DramFormat
#include "arch/dram/conv_shape.hpp"      // ConvShape (KH,KW,IC)
#include "arch/filter_buffer.hpp"        // FilterBuffer::kRowBytes

namespace sf { namespace dram {

/**
 * WeightLoader
 *
 * Loads all rows for a given oc_group from the layer's weights range and packs
 * them into a contiguous staging buffer, then writes them into FilterBuffer.
 *
 * SegmentHeader usage (as established):
 *  - kind              = SEG_WEIGHT
 *  - layer_id          = this layer
 *  - logical_spine_id  = flattened kk-index = ky * KW + kx   (recommended)
 *  - size              = 128  (number of int8 weights)
 *  - aux0              = inC
 *  - aux1              = oc_group
 *  - eol/seg_id/count  = per-stream segmentation if needed (not relied upon)
 *
 * Row ordering inside staging (local to the current oc_group):
 *   local_row = (inC * KH + ky) * KW + kx
 * i.e., inC-major, then ky, then kx. This is deterministic and compact.
 */
class WeightLoader {
public:
  // WeightLoader does not own fmt/img/dir; they must outlive this object.
  WeightLoader(const DramFormat&         weight_fmt,
               const DramImage&          img,
               const LayerDirectory&     dir,
               std::uint16_t             layer_id,
               ConvShape                 shape)
  : fmt_(&weight_fmt), reader_(weight_fmt, img, dir), layer_id_(layer_id), shape_(shape) {}

  /**
   * Load an entire oc_group into the provided FilterBuffer.
   *
   * Returns true if at least one row was loaded.
   * Throws std::invalid_argument on shape mismatch or malformed segments.
   */
  bool LoadOCGroupTo(FilterBuffer& fb, std::uint16_t oc_group);

private:
  // Validate indices are within shape bounds.
  inline void check_indices(std::uint16_t inC, std::uint16_t ky, std::uint16_t kx) const {
    if (inC >= shape_.IC) throw std::invalid_argument("WeightLoader: inC out of range");
    if (ky  >= shape_.KH) throw std::invalid_argument("WeightLoader: ky out of range");
    if (kx  >= shape_.KW) throw std::invalid_argument("WeightLoader: kx out of range");
  }

  // Compute local row index for this oc_group.
  inline std::size_t local_row(std::uint16_t inC, std::uint16_t ky, std::uint16_t kx) const {
    return (static_cast<std::size_t>(inC) * shape_.KH + ky) * shape_.KW + kx;
  }

private:
  const DramFormat* fmt_;     // weight format (entry_bytes=1, maxEntries=128)
  StreamReader      reader_;  // reader bound to weights range (constructed with weight_fmt)
  std::uint16_t     layer_id_;
  ConvShape         shape_;
};

}} // namespace sf::dram
