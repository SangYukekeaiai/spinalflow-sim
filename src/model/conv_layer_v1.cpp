// All comments are in English.

#include "model/conv_layer.hpp"
#include <algorithm>

namespace sf {

void ConvLayer::ConfigureLayer(int layer_id,
                               int C_in, int C_out,
                               int H_in, int W_in,
                               int Kh, int Kw,
                               int Sh, int Sw,
                               int Ph, int Pw,
                               sf::dram::SimpleDRAM* dram)
{
  // Stash parameters
  layer_id_ = layer_id;
  C_in_ = C_in;   C_out_ = C_out;
  H_in_ = H_in;   W_in_  = W_in;
  Kh_ = Kh; Kw_ = Kw;
  Sh_ = Sh; Sw_ = Sw;
  Ph_ = Ph; Pw_ = Pw;

  // Derive output spatial dimensions.
  H_out_ = DeriveOutDim(H_in_, Ph_, Kh_, Sh_);
  W_out_ = DeriveOutDim(W_in_, Pw_, Kw_, Sw_);

  // Configure FilterBuffer with layer-wide static params and DRAM.
  // FB is shared with Core (Core uses it during PEArray.run()).
  fb_.Configure(C_in_, W_in_, Kh_, Kw_, Sh_, Sw_, Ph_, Pw_, dram);

  // Validate tile divisibility and capacity here (fail fast).
  if (C_out_ <= 0 || (C_out_ % static_cast<int>(kNumPE)) != 0) {
    throw std::invalid_argument("ConvLayer::ConfigureLayer: C_out must be positive and divisible by kNumPE.");
  }
  const int total_tiles = C_out_ / static_cast<int>(kNumPE);
  if (total_tiles > static_cast<int>(kTilesPerSpine)) {
    throw std::invalid_argument("ConvLayer::ConfigureLayer: total_tiles exceeds kTilesPerSpine.");
  }
}

std::vector<std::vector<int>> ConvLayer::generate_batches(int h_out, int w_out) const {
  // 1) Enumerate kernel window and back-project to input coordinates.
  std::vector<int> spine_ids;
  spine_ids.reserve(static_cast<std::size_t>(Kh_ * Kw_));

  for (int r = 0; r < Kh_; ++r) {
    for (int c = 0; c < Kw_; ++c) {
      const int h_in = h_out * Sh_ - Ph_ + r;
      const int w_in = w_out * Sw_ - Pw_ + c;
      if (h_in < 0 || h_in >= H_in_ || w_in < 0 || w_in >= W_in_) {
        // Out-of-bounds due to padding; skip.
        continue;
      }
      const int spine_id = h_in * W_in_ + w_in;
      spine_ids.push_back(spine_id);
    }
  }

  // 2) batches_needed per spec: ceil(Kh * Kw / kNumPhysISB).
  const int kernel_slots = Kh_ * Kw_;
  const int batches_needed = (kernel_slots + static_cast<int>(kNumPhysISB) - 1) / static_cast<int>(kNumPhysISB);
  const int bn = std::max(1, batches_needed);

  // 3) Partition spine_ids into bn batches in-order (tail batch may be shorter).
  std::vector<std::vector<int>> batches;
  batches.resize(static_cast<std::size_t>(bn));

  // Distribute linearly chunked by kNumPhysISB (consistent with ISB fan-in).
  std::size_t cursor = 0;
  for (int b = 0; b < bn; ++b) {
    const std::size_t take = std::min<std::size_t>(kNumPhysISB, spine_ids.size() - cursor);
    if (take == 0) {
      // Allow empty tail batches if many kernel slots were OOB.
      continue;
    }
    batches[static_cast<std::size_t>(b)].insert(
      batches[static_cast<std::size_t>(b)].end(),
      spine_ids.begin() + static_cast<std::ptrdiff_t>(cursor),
      spine_ids.begin() + static_cast<std::ptrdiff_t>(cursor + take)
    );
    cursor += take;
  }

  return batches;
}

void ConvLayer::run_layer() {
  // For each output site (h_out, w_out)
  for (int h = 0; h < H_out_; ++h) {
    for (int w = 0; w < W_out_; ++w) {
      // Step A: Update FilterBuffer with the current site.
      fb_.Update(h, w);

      // Step B: Bind Core site context (also sets OutputSpine's spine_id).
      core_.SetSpineContext(layer_id_, h, w, W_out_);

      // Step C: Configure tiles for this site; clears TOB buffers for a clean start.
      core_.ConfigureTiles(C_out_);

      // Step D: Build batches for this site and bind to Core.
      auto batches = generate_batches(h, w);
      core_.BindTileBatches(&batches);

      // Step E: Preload first batch into ISB.
      core_.PreloadFirstBatch();

      // Step F: Per-tile compute (no global drain/store here).
      const int total_tiles = core_.total_tiles();
      for (int tile_id = 0; tile_id < total_tiles; ++tile_id) {
        // Load weight tile for this tile_id.
        fb_.LoadWeightFromDram(static_cast<std::uint32_t>(layer_id_),
                               static_cast<std::uint32_t>(tile_id));

        // Drive the pipeline until compute finishes for this tile.
        while (!core_.FinishedCompute()) {
          core_.StepOnce(tile_id);
        }
      }

      // Step G: After all tiles have finished, perform global drain + store once.
      core_.DrainAllTilesAndStore();
    }
  }
}

} // namespace sf
