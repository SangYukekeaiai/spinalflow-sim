// All comments are in English.
#include "model/fc_layer.hpp"
#include <iostream>
namespace sf {

void FCLayer::EnsureEngines_(sf::dram::SimpleDRAM* dram) {
  if (!fb_)  fb_  = std::make_unique<FilterBuffer>();
  isb_  = std::make_unique<InputSpineBuffer>(dram);
  core_ = std::make_unique<Core>(dram, fb_.get(), isb_.get());
}

void FCLayer::ConfigureLayer(int layer_id,
                             int C_in, int C_out,
                             int H_in, int W_in,
                             int Kh, int Kw,
                             int Sh, int Sw,
                             int Ph, int Pw,
                             int Threshold,
                             sf::dram::SimpleDRAM* dram)
{
  layer_id_ = layer_id;
  C_in_ = C_in;   C_out_ = C_out;
  H_in_ = H_in;   W_in_  = W_in;
  Kh_ = Kh; Kw_ = Kw;
  Sh_ = Sh; Sw_ = Sw;
  Ph_ = Ph; Pw_ = Pw;

  H_out_ = DeriveOutDim(H_in_, Ph_, Kh_, Sh_);
  W_out_ = DeriveOutDim(W_in_, Pw_, Kw_, Sw_);

  threshold_ = Threshold;

  EnsureEngines_(dram);
  fb_->Configure(C_in_, W_in_, Kh_, Kw_, Sh_, Sw_, Ph_, Pw_, dram);

  if (C_out_ <= 0) {
    throw std::invalid_argument("FCLayer::ConfigureLayer: C_out must be positive and divisible by kNumPE.");
  }
  const int total_tiles = (C_out_ + static_cast<int>(kNumPE)) / static_cast<int>(kNumPE);
  if (total_tiles > static_cast<int>(kTilesPerSpine)) {
    throw std::invalid_argument("FCLayer::ConfigureLayer: total_tiles exceeds kTilesPerSpine.");
  }
}

std::vector<std::vector<int>> FCLayer::generate_batches(int /*h_out*/, int /*w_out*/) const {
  std::vector<int> spine_ids;
  spine_ids.reserve(static_cast<std::size_t>(H_in_ * W_in_));

  for (int h = 0; h < H_in_; ++h) {
    for (int w = 0; w < W_in_; ++w) {
      spine_ids.push_back(h * W_in_ + w);
    }
  }

  const int total_slots    = H_in_ * W_in_;
  const int batches_needed = (total_slots + static_cast<int>(kNumPhysISB) - 1) / static_cast<int>(kNumPhysISB);
  const int bn = std::max(1, batches_needed);

  std::vector<std::vector<int>> batches(static_cast<std::size_t>(bn));
  std::size_t cursor = 0;
  for (int b = 0; b < bn; ++b) {
    const std::size_t take = std::min<std::size_t>(kNumPhysISB, spine_ids.size() - cursor);
    if (take == 0) continue;
    batches[static_cast<std::size_t>(b)].insert(
      batches[static_cast<std::size_t>(b)].end(),
      spine_ids.begin() + static_cast<std::ptrdiff_t>(cursor),
      spine_ids.begin() + static_cast<std::ptrdiff_t>(cursor + take)
    );
    cursor += take;
  }
  return batches;
}

void FCLayer::run_layer() {
  std::cout << "FCLayer::run_layer: Running layer " << layer_id_ << " with output shape (" << H_out_ << ", " << W_out_ << ")\n";
  if (!core_ || !fb_) throw std::runtime_error("FCLayer::run_layer: engines not configured.");

  for (int h = 0; h < H_out_; ++h) {
    for (int w = 0; w < W_out_; ++w) {
      fb_->Update(h, w);
      core_->SetSpineContext(layer_id_, h, w, W_out_);
      core_->ConfigureTiles(C_out_);

      auto batches = generate_batches(h, w);
      core_->BindTileBatches(&batches);
      core_->PreloadFirstBatch();

      const int total_tiles = core_->total_tiles();
      for (int tile_id = 0; tile_id < total_tiles; ++tile_id) {
        fb_->LoadWeightFromDram(static_cast<std::uint32_t>(layer_id_),
                                static_cast<std::uint32_t>(tile_id));
        core_->InitPEsBeforeLoop(threshold_, tile_id); // threshold=1 for now
        while (!core_->FinishedCompute()) {
          core_->StepOnce(tile_id);
        }
      }
      core_->DrainAllTilesAndStore();
    }
  }
}

} // namespace sf
