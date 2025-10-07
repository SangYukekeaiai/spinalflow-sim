// All comments are in English.
#include "simulation.hpp"
#include <fstream>
#include <iterator>
#include <algorithm>

using nlohmann::json;

namespace sf {

static LayerKind ParseKind_(const std::string& s) {
  if (s == "conv") return LayerKind::kConv;
  if (s == "fc")   return LayerKind::kFC;
  throw std::invalid_argument("Unknown layer kind: " + s);
}

std::vector<LayerSpec> ParseConfig(const std::string& json_path) {
  // Read entire JSON file as text
  std::ifstream ifs(json_path);
  if (!ifs) throw std::runtime_error("ParseConfig: cannot open json file: " + json_path);
  std::string jtxt((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());

  // Parse json
  json j = json::parse(jtxt);
  if (!j.contains("layers") || !j["layers"].is_array()) {
    throw std::invalid_argument("ParseConfig: missing 'layers' array");
  }

  std::vector<LayerSpec> out;
  out.reserve(j["layers"].size());

  for (const auto& jl : j["layers"]) {
    LayerSpec s;

    if (!jl.contains("L")) throw std::invalid_argument("ParseConfig: layer entry missing 'L'");
    s.L = jl.at("L").get<int>();

    s.name = jl.value("name", std::string("L") + std::to_string(s.L));
    s.kind = ParseKind_(jl.at("kind").get<std::string>());

    // params_in
    {
      const auto& pin = jl.at("params_in");
      s.Cin_in = pin.at("C").get<int>();
      s.H_in   = pin.at("H").get<int>();
      s.W_in   = pin.at("W").get<int>();
    }

    // params_weight (with dilation)
    {
      const auto& pw = jl.at("params_weight");
      s.Cin_w = pw.at("Cin").get<int>();
      s.Cout  = pw.at("Cout").get<int>();
      s.Kh    = pw.at("Kh").get<int>();
      s.Kw    = pw.at("Kw").get<int>();

      const auto& stride = pw.at("stride");
      s.Sh = stride.at("h").get<int>();
      s.Sw = stride.at("w").get<int>();

      const auto& pad = pw.at("padding");
      s.Ph = pad.at("h").get<int>();
      s.Pw = pad.at("w").get<int>();

      const auto& dil = pw.at("dilation");
      s.Dh = dil.at("h").get<int>();
      s.Dw = dil.at("w").get<int>();

      // For now, we only support dilation = 1
      if (s.Dh != 1 || s.Dw != 1) {
        throw std::invalid_argument("ParseConfig: dilation != 1 is not supported yet.");
      }
    }

    // params_out (optional for checking)
    if (jl.contains("params_out")) {
      const auto& po = jl.at("params_out");
      s.Cout_out = po.at("C").get<int>();
      s.H_out    = po.at("H").get<int>();
      s.W_out    = po.at("W").get<int>();
    }

    // Basic sanity checks to fail fast
    if (s.Cin_in != s.Cin_w) {
      throw std::invalid_argument("ParseConfig: Cin mismatch between params_in.C and params_weight.Cin at L=" + std::to_string(s.L));
    }
    if (s.Cout <= 0) {
      throw std::invalid_argument("ParseConfig: Cout must be positive at L=" + std::to_string(s.L));
    }

    out.push_back(s);
  }

  // Keep layers ordered by L ascending just in case
  std::sort(out.begin(), out.end(), [](const LayerSpec& a, const LayerSpec& b){ return a.L < b.L; });
  return out;
}

sf::dram::SimpleDRAM InitDram(const std::string& bin_path, const std::string& json_path) {
  // Delegate to the convenience factory; it also builds per-layer tables.
  return sf::dram::SimpleDRAM::FromFiles(bin_path, json_path);
}

void RunNetwork(const std::vector<LayerSpec>& specs, sf::dram::SimpleDRAM* dram) {
  if (!dram) throw std::invalid_argument("RunNetwork: null DRAM pointer");

  for (const auto& s : specs) {
    if (s.kind == LayerKind::kConv) {
      ConvLayer conv;
      conv.ConfigureLayer(s.L,
                          s.Cin_in, s.Cout,
                          s.H_in,   s.W_in,
                          s.Kh,     s.Kw,
                          s.Sh,     s.Sw,
                          s.Ph,     s.Pw,
                          dram);
      conv.run_layer();
    } else if (s.kind == LayerKind::kFC) {
      FCLayer fc;
      fc.ConfigureLayer(s.L,
                        s.Cin_in, s.Cout,
                        s.H_in,   s.W_in,
                        s.Kh,     s.Kw,
                        s.Sh,     s.Sw,
                        s.Ph,     s.Pw,
                        dram);
      fc.run_layer();
    } else {
      throw std::runtime_error("RunNetwork: unsupported layer kind at L=" + std::to_string(s.L));
    }
  }
}



} // namespace sf
