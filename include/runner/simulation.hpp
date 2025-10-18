// All comments are in English.
#pragma once
#include <string>
#include <vector>
#include <stdexcept>
#include <cstdint>

#include <nlohmann/json.hpp>
#include "arch/dram/simple_dram.hpp"
#include "arch/cache/cache.hpp"
#include "model/conv_layer.hpp"
#include "model/fc_layer.hpp"

namespace sf {

enum class LayerKind { kConv, kFC };

struct LayerSpec {
  int        L = 0;
  LayerKind  kind = LayerKind::kConv;

  // Input
  int Cin_in = 0, H_in = 0, W_in = 0;

  // Weights / conv params
  int Cin_w = 0, Cout = 0, Kh = 1, Kw = 1;
  int Sh = 1, Sw = 1, Ph = 0, Pw = 0, Dh = 1, Dw = 1;

  // Optional output shape (for checking)
  int Cout_out = 0, H_out = 0, W_out = 0;

  // Misc
  float       threshold_ = 0.0f;
  std::string name;

  // ---- Minimal quantization metadata for weights ----
  int   w_bits       = 8;
  bool  w_signed     = true;
  int   w_frac_bits  = -1;   // n in Qm.n; -1 means "not provided"
  float w_scale      = 1.0f; // equals 2^-n for fixed-point

  // Optional debug provenance
  float w_float_min  = 0.0f;
  float w_float_max  = 0.0f;
  bool  has_w_qformat = false;
  bool  has_w_scale   = false;
};

std::vector<LayerSpec> ParseConfig(const std::string& json_path);
sf::dram::SimpleDRAM InitDram(const std::string& bin_path, const std::string& json_path);

void RunNetworkWithCacheOptions(const std::vector<LayerSpec>& specs,
                                sf::dram::SimpleDRAM* dram,
                                const std::string& repo_name,
                                const std::string& model_name,
                                const std::vector<std::size_t>& cache_sizes_bytes,
                                const std::vector<int>& cache_way_options,
                                const std::vector<int>& prefetch_depth_options,
                                const std::vector<sf::arch::cache::EvictionPolicy>& policies);

// Layers own their engines; RunNetwork simply configures and runs them.
void RunNetwork(const std::vector<LayerSpec>& specs,
                sf::dram::SimpleDRAM* dram,
                const std::string& repo_name,
                const std::string& model_name);

} // namespace sf
