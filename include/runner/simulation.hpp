// All comments are in English.
#pragma once
#include <string>
#include <vector>
#include <stdexcept>
#include <cstdint>

#include <nlohmann/json.hpp>
#include "arch/dram/simple_dram.hpp"
#include "model/conv_layer.hpp"
#include "model/fc_layer.hpp"

namespace sf {

enum class LayerKind { kConv, kFC };

struct LayerSpec {
  int        L = 0;
  LayerKind  kind = LayerKind::kConv;
  int Cin_in = 0, H_in = 0, W_in = 0;
  int Cin_w = 0, Cout = 0, Kh = 1, Kw = 1;
  int Sh = 1, Sw = 1, Ph = 0, Pw = 0, Dh = 1, Dw = 1;
  int Cout_out = 0, H_out = 0, W_out = 0;
  int threshold_ = 0;
  std::string name;
};

std::vector<LayerSpec> ParseConfig(const std::string& json_path);
sf::dram::SimpleDRAM InitDram(const std::string& bin_path, const std::string& json_path);

// NEW: no external engines needed; layers own engines.
void RunNetwork(const std::vector<LayerSpec>& specs, sf::dram::SimpleDRAM* dram);

} // namespace sf
