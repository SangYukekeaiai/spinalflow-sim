// simple_dram.cpp
// All comments are in English.
#include "arch/dram/simple_dram.hpp"
#include <fstream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <iostream>  // NEW

using nlohmann::json;
using sf::dram::SimpleDRAM;
using sf::dram::LayerMeta;
using sf::dram::SpineMeta;
using sf::dram::WeightTileMeta;

namespace sf { namespace dram {

void SimpleDRAM::LoadRawImage(const void* src, uint64_t n) {
  if (!src && n > 0) throw std::invalid_argument("LoadRawImage: null src with n>0");
  if (n > mem_.size()) throw std::out_of_range("LoadRawImage: image larger than DRAM capacity");
  if (n > 0) {
    std::memcpy(mem_.data(), src, static_cast<size_t>(n));
  }
}

void SimpleDRAM::BuildFromJson(const std::string& json_text) {
  json j = json::parse(json_text);

  if (!j.contains("layers") || !j["layers"].is_array()) {
    throw std::invalid_argument("BuildFromJson: missing 'layers' array");
  }

  for (const auto& jl : j["layers"]) {
    LayerMeta meta;

    // Input spines table
    if (jl.contains("input_spines")) {
      const auto& isp = jl["input_spines"];
      for (auto it = isp.begin(); it != isp.end(); ++it) {
        uint32_t spine_id = static_cast<uint32_t>(std::stoul(it.key()));
        const auto& v = it.value();
        SpineMeta sm;
        sm.id   = spine_id;
        sm.addr = static_cast<uint64_t>(v.at("addr").get<uint64_t>());
        sm.size = static_cast<uint32_t>(v.at("size").get<uint64_t>());
        meta.input_spines[spine_id] = sm;
      }
    }

    // Weight tiles table
    if (jl.contains("weight_tiles")) {
      const auto& wts = jl["weight_tiles"];
      for (auto it = wts.begin(); it != wts.end(); ++it) {
        uint32_t tile_id = static_cast<uint32_t>(std::stoul(it.key()));
        const auto& v = it.value();
        WeightTileMeta wm;
        wm.tile = tile_id;
        wm.addr = static_cast<uint64_t>(v.at("addr").get<uint64_t>());
        wm.size = static_cast<uint32_t>(v.at("size").get<uint64_t>());
        meta.weight_tiles[tile_id] = wm;
      }
    }

    // Optional output reservation
    meta.output_region_begin = jl.value("output_region_begin", 0ULL);
    meta.output_region_end   = jl.value("output_region_end",   0ULL);
    meta.output_write_ptr    = jl.value("output_write_ptr",    meta.output_region_begin);

    // Install meta for layer id L
    if (!jl.contains("L")) throw std::invalid_argument("BuildFromJson: layer entry missing 'L'");
    uint32_t L = jl["L"].get<uint32_t>();
    SetLayerMeta(L, std::move(meta));
  }
}

static std::vector<uint8_t> ReadAllBinary_(const std::string& path) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) throw std::runtime_error("FromFiles: cannot open bin file: " + path);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(ifs)),
                               std::istreambuf_iterator<char>());
}

static std::string ReadAllText_(const std::string& path) {
  std::ifstream ifs(path);
  if (!ifs) throw std::runtime_error("FromFiles: cannot open json file: " + path);
  return std::string((std::istreambuf_iterator<char>(ifs)),
                      std::istreambuf_iterator<char>());
}

SimpleDRAM SimpleDRAM::FromFiles(const std::string& bin_path, const std::string& json_path) {
  // Read files
  auto bin = ReadAllBinary_(bin_path);
  auto jtxt = ReadAllText_(json_path);

  // Create DRAM with exact size and load image
  SimpleDRAM dram(static_cast<uint64_t>(bin.size()));
  dram.LoadRawImage(bin.data(), static_cast<uint64_t>(bin.size()));

  // Build layer meta
  dram.BuildFromJson(jtxt);
  return dram;
}
}} // namespace sf::dram
