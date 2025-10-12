#pragma once

#include <vector>
#include <cstdint>
#include <stdexcept>
#include <algorithm>
#include <memory>
#include <unordered_map>

#include "common/constants.hpp"
#include "arch/dram/simple_dram.hpp"
#include "core/core.hpp"

namespace sf {

class ConvLayer {
public:
  ConvLayer() = default;

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

  // Builds batches for a single (h_out, w_out).
  std::vector<std::vector<int>> generate_batches(int h_out, int w_out) const;

  void run_layer();
private:
  static int DeriveOutDim(int in, int pad, int kernel, int stride) {
    const int numer = in + 2 * pad - kernel;
    if (numer < 0 || stride <= 0) {
      throw std::invalid_argument("ConvLayer: invalid shape for output dimension derivation.");
    }
    return numer / stride + 1;
  }


  static std::uint64_t PackHW(int h, int w) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(h)) << 32) |
           static_cast<std::uint32_t>(w);
  }

private:
  // --- Static layer parameters (immutable after ConfigureLayer) ---
  int layer_id_ = 0;

  int C_in_ = 0, C_out_ = 0;
  int H_in_ = 0, W_in_ = 0;
  int H_out_ = 0, W_out_ = 0;

  int Kh_ = 0, Kw_ = 0;
  int Sh_ = 0, Sw_ = 0;
  int Ph_ = 0, Pw_ = 0;

  float threshold_   = 0.0f;
  int   w_bits_      = 8;
  bool  w_signed_    = true;
  int   w_frac_bits_ = -1;
  float w_scale_     = 1.0f;

  // --- Derived/static per-layer quantities ---
  int batch_needed_ = 0;   // ceil((Kh*Kw)/kNumPhysISB)
  int total_tiles_  = 0;   // ceil(C_out/kNumPE)

  // --- Precomputed per-site batches ---
  // (h,w) -> batches (each batch is a vector of spine IDs)
  std::unordered_map<std::uint64_t, std::vector<std::vector<int>>> batches_per_hw_;

  // --- Runtime handles ---
  sf::dram::SimpleDRAM* dram_ = nullptr;     // non-owning
  std::unique_ptr<Core> core_;               // Core owns its own FB/ISB/etc.
};

} // namespace sf

