// All comments are in English.
#pragma once

#include <vector>
#include <cstdint>
#include <stdexcept>
#include <algorithm>
#include <memory>

#include "common/constants.hpp"
#include "arch/dram/simple_dram.hpp"
#include "arch/filter_buffer.hpp"
#include "arch/input_spine_buffer.hpp"
#include "core/core.hpp"

namespace sf {

/**
 * FCLayer
 * Owns Core, FilterBuffer, and InputSpineBuffer.
 * For FC, output spatial is typically 1x1, but we keep generality.
 */
class FCLayer {
public:
  FCLayer() = default;

  void ConfigureLayer(int layer_id,
                      int C_in, int C_out,
                      int H_in, int W_in,
                      int Kh, int Kw,
                      int Sh, int Sw,
                      int Ph, int Pw,
                      float Threshold,
                      int  w_bits,
                      bool w_signed,
                      int  w_frac_bits,
                      float w_scale,
                      sf::dram::SimpleDRAM* dram);

  std::vector<std::vector<int>> generate_batches(int h_out, int w_out) const;

  void run_layer();

private:
  static int DeriveOutDim(int in, int pad, int kernel, int stride) {
    const int numer = in + 2 * pad - kernel;
    if (numer < 0) {
      throw std::invalid_argument("FCLayer: invalid shape for output dimension derivation.");
    }
    return numer / stride + 1;
  }

  void EnsureEngines_(sf::dram::SimpleDRAM* dram);

private:
  int layer_id_{0};
  int C_in_{0}, C_out_{0};
  int H_in_{0}, W_in_{0};
  int Kh_{1}, Kw_{1};
  int Sh_{1}, Sw_{1};
  int Ph_{0}, Pw_{0};
  int H_out_{0}, W_out_{0};
  float threshold_{0.0f};
  int  w_bits_{8};
  bool w_signed_{true};
  int  w_frac_bits_{-1}; // -1 means "not specified"
  float w_scale_{1.0f};

  std::unique_ptr<FilterBuffer>     fb_;
  std::unique_ptr<InputSpineBuffer> isb_;
  std::unique_ptr<Core>             core_;
};

} // namespace sf
