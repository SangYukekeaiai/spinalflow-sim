// All comments are in English.
#include "model/conv_layer.hpp"
#include <algorithm>
#include <iostream>
#include "stats/layer_summary_csv.hpp"   // NEW
#include "stats/sim_stats.hpp"

namespace sf {

void ConvLayer::ConfigureLayer(int layer_id,
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
                               sf::dram::SimpleDRAM* dram)
{
  // 1) Save static params
  layer_id_ = layer_id;
  C_in_ = C_in;   C_out_ = C_out;
  H_in_ = H_in;   W_in_  = W_in;
  Kh_ = Kh; Kw_ = Kw;
  Sh_ = Sh; Sw_ = Sw;
  Ph_ = Ph; Pw_ = Pw;
  threshold_   = Threshold;
  w_bits_      = w_bits;
  w_signed_    = w_signed;
  w_frac_bits_ = w_frac_bits;
  w_scale_     = w_scale;

  // 2) Derive output dims
  H_out_ = DeriveOutDim(H_in_, Ph_, Kh_, Sh_);
  W_out_ = DeriveOutDim(W_in_, Pw_, Kw_, Sw_);

  // 3) Compute per-layer tiles and batches
  if (C_out_ <= 0) {
    throw std::invalid_argument("ConvLayer::ConfigureLayer: C_out must be positive.");
  }
  total_tiles_ = static_cast<int>(
      (static_cast<long long>(C_out_) + static_cast<long long>(kNumPE) - 1LL) /
      static_cast<long long>(kNumPE));
  if (total_tiles_ <= 0 || total_tiles_ > static_cast<int>(kTilesPerSpine)) {
    throw std::invalid_argument("ConvLayer::ConfigureLayer: total_tiles out of range.");
  }

  const int kernel_slots = Kh_ * Kw_;
  batch_needed_ = (kernel_slots + static_cast<int>(kNumPhysISB) - 1) /
                  static_cast<int>(kNumPhysISB);
  if (batch_needed_ <= 0) batch_needed_ = 1;

  // 4) Precompute batches map for every (h,w)
  batches_per_hw_.clear();
  batches_per_hw_.reserve(static_cast<std::size_t>(H_out_) * static_cast<std::size_t>(W_out_));
  for (int h = 0; h < H_out_; ++h) {
    for (int w = 0; w < W_out_; ++w) {
      batches_per_hw_.emplace(PackHW(h, w), generate_batches(h, w));
    }
  }

  // 5) Hold DRAM and construct Core (value-members architecture)
  dram_ = dram;
  core_ = std::make_unique<Core>(
              dram_,
              layer_id_, C_in_, C_out_,
              H_in_, W_in_,
              H_out_, W_out_,
              Kh_, Kw_,
              Sh_, Sw_,
              Ph_, Pw_,
              threshold_,
              w_bits_, w_signed_, w_frac_bits_, w_scale_,
              total_tiles_,
              &batches_per_hw_,
              batch_needed_);
}

std::vector<std::vector<int>> ConvLayer::generate_batches(int h_out, int w_out) const {
  // Build valid input spine IDs for this (h_out, w_out).
  std::vector<int> spine_ids;
  spine_ids.reserve(static_cast<std::size_t>(Kh_ * Kw_));

  for (int r = 0; r < Kh_; ++r) {
    for (int c = 0; c < Kw_; ++c) {
      const int h_in = h_out * Sh_ - Ph_ + r;
      const int w_in = w_out * Sw_ - Pw_ + c;

      if (h_in < 0 || h_in >= H_in_ || w_in < 0 || w_in >= W_in_) continue;

      const int spine_id = h_in * W_in_ + w_in;
      spine_ids.push_back(spine_id);
    }
  }

  std::vector<std::vector<int>> batches;
  if (spine_ids.empty()) return batches;

  const std::size_t total = spine_ids.size();
  const std::size_t B = (total + static_cast<std::size_t>(kNumPhysISB) - 1) /
                        static_cast<std::size_t>(kNumPhysISB);
  batches.resize(B);

  std::size_t cursor = 0;
  for (std::size_t b = 0; b < B; ++b) {
    const std::size_t take = std::min<std::size_t>(kNumPhysISB, total - cursor);
    auto& dst = batches[b];
    dst.insert(dst.end(),
               spine_ids.begin() + static_cast<std::ptrdiff_t>(cursor),
               spine_ids.begin() + static_cast<std::ptrdiff_t>(cursor + take));
    cursor += take;
  }
  return batches;
}

// All comments are in English.
void ConvLayer::run_layer() {
  if (!core_) {
    throw std::runtime_error("ConvLayer::run_layer: core not configured.");
  }
  int unused = 0;
  for (int h = 0; h < H_out_; ++h) {
    for (int w = 0; w < W_out_; ++w) {
      // Per-output-spine preparation.
      core_->PrepareForSpine(h, w);

      // Iterate all tiles for this (h, w).
      const int total_tiles = core_->total_tiles();
      for (int tile_id = 0; tile_id < total_tiles; ++tile_id) {
        // Per-tile preparation and compute across all batches.
        core_->PrepareForTile(tile_id);
        core_->Compute_EachTile(tile_id);
      }

      // Drain tile buffers into OutputSpine and store to DRAM.
      
      core_->DrainAllTilesAndStore(unused);

    }
  }
  std::cout << "drained entries: " << unused / (H_out_ * W_out_) << "\n";
}


} // namespace sf
