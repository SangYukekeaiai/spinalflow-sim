// All comments are in English.
#pragma once

#include <vector>
#include <cstdint>
#include <stdexcept>
#include <algorithm>

#include "common/constants.hpp"
#include "arch/dram/simple_dram.hpp"
#include "arch/filter_buffer.hpp"
#include "core/core.hpp"

namespace sf {

/**
 * FCLayer
 * - Mirrors ConvLayer control flow.
 * - The only difference is batching: FC uses ALL input spines of the input map
 *   (size H_in * W_in) for its single spatial site.
 */
class FCLayer {
public:
  // Configure the layer with dimensions and DRAM handle.
  // For FC, upstream scripts will flatten inputs so typically:
  //   H_in = 1, W_in = 1, Kh = 1, Kw = 1, Sh = Sw = 1, Ph = Pw = 0.
  void ConfigureLayer(int layer_id,
                      int C_in, int C_out,
                      int H_in, int W_in,
                      int Kh, int Kw,
                      int Sh, int Sw,
                      int Ph, int Pw,
                      sf::dram::SimpleDRAM* dram);

  // Build batches of input spine ids for a given output site (h_out, w_out).
  // FC-specific: include ALL spines from the input feature map.
  std::vector<std::vector<int>> generate_batches(int h_out, int w_out) const;

  // Execute this layer end-to-end (per site, per tile).
  void run_layer();

private:
  static int DeriveOutDim(int in, int pad, int kernel, int stride) {
  // (in + 2*pad - kernel) must be divisible by stride
  const int numer = in + 2 * pad - kernel;
  if (numer < 0 || numer % stride != 0) {
    throw std::invalid_argument("FCLayer: invalid shape for output dimension derivation.");
  }
  return numer / stride + 1;
}

  // Parameters
  int layer_id_{0};
  int C_in_{0}, C_out_{0};
  int H_in_{0}, W_in_{0};
  int Kh_{1}, Kw_{1};
  int Sh_{1}, Sw_{1};
  int Ph_{0}, Pw_{0};

  // Derived output spatial size (often 1x1 for FC).
  int H_out_{0}, W_out_{0};

  // Shared components
  FilterBuffer fb_;
  Core         core_;
};

} // namespace sf
