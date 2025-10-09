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

class ConvLayer {
public:
  ConvLayer() = default;

  // Optional: pre-install custom engines (FilterBuffer/Core).
  // Note: Core will be re-created in ConfigureLayer because it needs DRAM+ISB.
  void SetEngines(std::unique_ptr<Core> core, std::unique_ptr<FilterBuffer> fb) {
    core_ = std::move(core);
    fb_   = std::move(fb);
  }

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
      throw std::invalid_argument("ConvLayer: invalid shape for output dimension derivation.");
    }
    return numer / stride + 1;
  }

  void EnsureEngines_(sf::dram::SimpleDRAM* dram);

private:
  // --- Layer parameters ---
  int layer_id_ = 0;
  int C_in_ = 0, C_out_ = 0;
  int H_in_ = 0, W_in_ = 0;
  int H_out_ = 0, W_out_ = 0;
  int Kh_ = 0, Kw_ = 0;
  int Sh_ = 0, Sw_ = 0;
  int Ph_ = 0, Pw_ = 0;
  float threshold_ = 0.0f;
  int   w_bits_      = 8;
  bool  w_signed_    = true;
  int   w_frac_bits_ = -1;    // -1 means "unknown"
  float w_scale_     = 1.0f;  // equals 2^-n when fixed-point is used

  // Owning engines
  std::unique_ptr<FilterBuffer>    fb_;
  std::unique_ptr<InputSpineBuffer> isb_;
  std::unique_ptr<Core>            core_;
};

} // namespace sf
