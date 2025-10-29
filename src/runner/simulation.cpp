// All comments are in English.
#include "runner/simulation.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <iostream>

#include "utils/stats_io.hpp"
#include "utils/stats_types.hpp"

using nlohmann::json;

namespace sf {

namespace {

LayerKind ParseKind(const std::string& s) {
  if (s == "conv") return LayerKind::kConv;
  if (s == "fc")   return LayerKind::kFC;
  throw std::invalid_argument("Unknown layer kind: " + s);
}

} // namespace

std::vector<LayerSpec> ParseConfig(const std::string& json_path) {
  std::ifstream ifs(json_path);
  if (!ifs) {
    throw std::runtime_error("ParseConfig: cannot open json file: " + json_path);
  }
  std::string jtxt((std::istreambuf_iterator<char>(ifs)),
                   std::istreambuf_iterator<char>());

  json j = json::parse(jtxt);
  if (!j.contains("layers") || !j["layers"].is_array()) {
    throw std::invalid_argument("ParseConfig: missing 'layers' array");
  }

  std::vector<LayerSpec> out;
  out.reserve(j["layers"].size());

  for (const auto& jl : j["layers"]) {
    LayerSpec s;

    if (!jl.contains("L")) {
      throw std::invalid_argument("ParseConfig: layer entry missing 'L'");
    }
    s.L = jl.at("L").get<int>();

    s.name       = jl.value("name", std::string("L") + std::to_string(s.L));
    s.kind       = ParseKind(jl.at("kind").get<std::string>());
    s.threshold_ = jl.value("threshold", 0.0f);

    {
      const auto& pin = jl.at("params_in");
      s.Cin_in = pin.at("C").get<int>();
      s.H_in   = pin.at("H").get<int>();
      s.W_in   = pin.at("W").get<int>();
    }

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
      if (s.Dh != 1 || s.Dw != 1) {
        throw std::invalid_argument("ParseConfig: dilation != 1 is not supported yet.");
      }
    }

    if (jl.contains("params_out")) {
      const auto& po = jl.at("params_out");
      s.Cout_out = po.at("C").get<int>();
      s.H_out    = po.at("H").get<int>();
      s.W_out    = po.at("W").get<int>();
    }

    if (jl.contains("weight_q_format") && jl["weight_q_format"].is_object()) {
      const auto& qf = jl["weight_q_format"];
      s.w_bits      = qf.value("bits", 8);
      s.w_signed    = qf.value("signed", true);
      s.w_frac_bits = qf.value("frac_bits", -1);
      s.has_w_qformat = true;
    }
    if (jl.contains("weight_scale")) {
      s.w_scale = jl.at("weight_scale").get<float>();
      s.has_w_scale = true;
    } else if (jl.contains("weight_qparams") && jl["weight_qparams"].is_object()) {
      const auto& qp = jl["weight_qparams"];
      if (qp.contains("scale")) {
        s.w_scale = qp.at("scale").get<float>();
        s.has_w_scale = true;
      }
    }
    if (jl.contains("weight_float_min")) s.w_float_min = jl.at("weight_float_min").get<float>();
    if (jl.contains("weight_float_max")) s.w_float_max = jl.at("weight_float_max").get<float>();

    if (s.has_w_qformat && s.has_w_scale && s.w_frac_bits >= 0) {
      const float expect = std::ldexp(1.0f, -s.w_frac_bits);
      const float eps = 1e-6f * std::max(1.0f, std::abs(expect));
      if (std::abs(s.w_scale - expect) > eps) {
        std::cerr << "[ParseConfig][Warn] L=" << s.L
                  << " weight_scale (" << s.w_scale
                  << ") != 2^-frac_bits (" << expect
                  << "). Proceeding with provided values.\n";
      }
    }

    if (s.Cin_in != s.Cin_w) {
      throw std::invalid_argument("ParseConfig: Cin mismatch between params_in.C and params_weight.Cin at L=" +
                                  std::to_string(s.L));
    }
    if (s.Cout <= 0) {
      throw std::invalid_argument("ParseConfig: Cout must be positive at L=" + std::to_string(s.L));
    }

    out.push_back(s);
  }

  std::sort(out.begin(), out.end(),
            [](const LayerSpec& a, const LayerSpec& b) { return a.L < b.L; });
  return out;
}

sf::dram::SimpleDRAM InitDram(const std::string& bin_path, const std::string& json_path) {
  return sf::dram::SimpleDRAM::FromFiles(bin_path, json_path);
}

void RunNetwork(const std::vector<LayerSpec>& specs,
                sf::dram::SimpleDRAM* dram,
                const std::string& repo_name,
                const std::string& model_name,
                bool write_stats_csv) {
  if (!dram) {
    throw std::invalid_argument("RunNetwork: dram pointer is null");
  }

  std::vector<LayerStageRecord> stage_rows;
  stage_rows.reserve(specs.size());

  for (const auto& spec : specs) {
    switch (spec.kind) {
      case LayerKind::kConv: {
        ConvLayer layer;
        layer.ConfigureLayer(spec.L,
                             spec.Cin_in, spec.Cout,
                             spec.H_in,   spec.W_in,
                             spec.Kh,     spec.Kw,
                             spec.Sh,     spec.Sw,
                             spec.Ph,     spec.Pw,
                             spec.threshold_,
                             spec.w_bits,
                             spec.w_signed,
                             spec.w_frac_bits,
                             spec.w_scale,
                             dram);
        layer.run_layer();
        stage_rows.push_back(LayerStageRecord{
            spec.L,
            spec.name,
            spec.kind,
            layer.cycle_stats(),
            layer.sram_stats()
        });
        break;
      }
      case LayerKind::kFC: {
        FCLayer layer;
        layer.ConfigureLayer(spec.L,
                             spec.Cin_in, spec.Cout,
                             spec.H_in,   spec.W_in,
                             spec.Kh,     spec.Kw,
                             spec.Sh,     spec.Sw,
                             spec.Ph,     spec.Pw,
                             spec.threshold_,
                             spec.w_bits,
                             spec.w_signed,
                             spec.w_frac_bits,
                             spec.w_scale,
                             dram);
        layer.run_layer();
        stage_rows.push_back(LayerStageRecord{
            spec.L,
            spec.name,
            spec.kind,
            layer.cycle_stats(),
            layer.sram_stats()
        });
        break;
      }
      default:
        throw std::runtime_error("RunNetwork: unsupported layer kind");
    }
  }

  if (write_stats_csv) {
    sf::WriteStageCyclesCsv(repo_name, model_name, stage_rows);
  }
}

} // namespace sf
