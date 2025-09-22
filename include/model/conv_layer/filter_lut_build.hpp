#pragma once
#include <cstdint>
#include "driver/weight_lut.hpp"

namespace sf {
namespace model {

/**
 * BuildFilterLUT
 *  - Thin wrapper to construct a driver::WeightLUT for a convolution layer.
 *  - The LUT encodes mapping (ky,kx,in_c,out_tile) -> FilterBuffer row_id,
 *    and a stable neuron_id for (ky,kx,in_c).
 *
 * Inputs:
 *   inC, outC:    layer channels
 *   kH, kW:       kernel size
 *
 * Notes:
 *   - ox, oy, strideW/H, padW/H, inW/H are NOT required to build the weight LUT,
 *     because the LUT only depends on kernel geometry and channels.
 */
inline driver::WeightLUT BuildFilterLUT(
    uint16_t inC, uint16_t outC, uint8_t kH, uint8_t kW) {
  driver::WeightLUT lut;
  driver::WeightLUT::Params p;
  p.inC = inC;
  p.outC = outC;
  p.kH = kH;
  p.kW = kW;
  p.peLanes = 128; // must match FilterBuffer row width
  lut.Build(p);
  return lut;
}

} // namespace model
} // namespace sf
