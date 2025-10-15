// All comments are in English.
#include "runner/simulation.hpp"
#include <fstream>
#include <iterator>
#include <algorithm>
#include <cmath>     // for std::ldexp, std::abs
#include <filesystem>
#include <cctype>
#include <iomanip>

using nlohmann::json;

namespace sf {

namespace {

struct LayerStageRecord {
  int layer_id = 0;
  std::string layer_name;
  LayerKind kind = LayerKind::kConv;
  CoreCycleStats cycles{};
  CoreSramStats sram_stats{};
};

std::string SanitizeName(const std::string& input) {
  std::string out;
  out.reserve(input.size());
  for (char ch : input) {
    const unsigned char uch = static_cast<unsigned char>(ch);
    if (std::isalnum(uch) || ch == '_' || ch == '-') {
      out.push_back(ch);
    } else {
      out.push_back('_');
    }
  }
  if (out.empty()) {
    out = "unnamed";
  }
  return out;
}

const char* LayerKindToString(LayerKind kind) {
  switch (kind) {
    case LayerKind::kConv: return "conv";
    case LayerKind::kFC:   return "fc";
    default:               return "unknown";
  }
}

std::filesystem::path BuildSramAccessCsvPath(const std::string& repo_name,
                                             const std::string& model_name) {
  const auto sanitized_repo  = SanitizeName(repo_name);
  const auto sanitized_model = SanitizeName(model_name);
  std::filesystem::path dir("stats");
  std::filesystem::path file =
      sanitized_repo + "__" + sanitized_model + "__sram_access.csv";
  return dir / file;
}

std::filesystem::path BuildSramCapacityCsvPath(const std::string& repo_name,
                                               const std::string& model_name) {
  const auto sanitized_repo  = SanitizeName(repo_name);
  const auto sanitized_model = SanitizeName(model_name);
  std::filesystem::path dir("stats");
  std::filesystem::path file =
      sanitized_repo + "__" + sanitized_model + "__sram_capacity.csv";
  return dir / file;
}

std::filesystem::path BuildStageCsvPath(const std::string& repo_name,
                                        const std::string& model_name) {
  const auto sanitized_repo  = SanitizeName(repo_name);
  const auto sanitized_model = SanitizeName(model_name);
  std::filesystem::path dir("stats");
  std::filesystem::path file =
      sanitized_repo + "__" + sanitized_model + "__stage_cycles.csv";
  return dir / file;
}

std::filesystem::path BuildLayerTablesDir(const std::string& repo_name,
                                          const std::string& model_name) {
  const auto sanitized_repo  = SanitizeName(repo_name);
  const auto sanitized_model = SanitizeName(model_name);
  std::filesystem::path dir("stats");
  dir /= sanitized_repo + "__" + sanitized_model;
  return dir;
}

void WriteStageCyclesCsv(const std::string& repo_name,
                         const std::string& model_name,
                         const std::vector<LayerStageRecord>& rows) {
  const auto csv_path = BuildStageCsvPath(repo_name, model_name);
  std::filesystem::create_directories(csv_path.parent_path());
  std::ofstream ofs(csv_path, std::ios::out | std::ios::trunc);
  if (!ofs) {
    throw std::runtime_error("RunNetwork: failed to open stage cycles CSV file " + csv_path.string());
  }

  ofs << "repo,model,layer_id,layer_name,layer_kind,load_cycles,compute_cycles,store_cycles\n";
  for (const auto& row : rows) {
    ofs << repo_name << ','
        << model_name << ','
        << row.layer_id << ','
        << std::quoted(row.layer_name) << ','
        << LayerKindToString(row.kind) << ','
        << row.cycles.load_cycles << ','
        << row.cycles.compute_cycles << ','
        << row.cycles.store_cycles << '\n';
  }
  ofs.flush();
  std::cout << "[Simulation] Stage cycles CSV written to " << csv_path << "\n";
}

void WriteSramAccessCsv(const std::string& repo_name,
                        const std::string& model_name,
                        const std::vector<LayerStageRecord>& rows) {
  const auto csv_path = BuildSramAccessCsvPath(repo_name, model_name);
  std::filesystem::create_directories(csv_path.parent_path());
  std::ofstream ofs(csv_path, std::ios::out | std::ios::trunc);
  if (!ofs) {
    throw std::runtime_error("RunNetwork: failed to open SRAM access CSV file " + csv_path.string());
  }

  ofs << "model,layer_id,layer_name,layer_kind,"
         "isb_accesses,filter_accesses,output_accesses,total_cycles\n";
  for (const auto& row : rows) {
    const std::uint64_t total_cycles =
        row.cycles.load_cycles + row.cycles.compute_cycles + row.cycles.store_cycles;
    ofs << model_name << ','
        << row.layer_id << ','
        << std::quoted(row.layer_name) << ','
        << LayerKindToString(row.kind) << ','
        << row.sram_stats.input_spine.accesses << ','
        << row.sram_stats.filter.accesses << ','
        << row.sram_stats.output_spine.accesses << ','
        << total_cycles << '\n';
  }
  ofs.flush();
  std::cout << "[Simulation] SRAM access CSV written to " << csv_path << "\n";
}

void WriteSramCapacityCsv(const std::string& repo_name,
                          const std::string& model_name,
                          const std::vector<LayerStageRecord>& rows) {
  const auto csv_path = BuildSramCapacityCsvPath(repo_name, model_name);
  std::filesystem::create_directories(csv_path.parent_path());
  std::ofstream ofs(csv_path, std::ios::out | std::ios::trunc);
  if (!ofs) {
    throw std::runtime_error("RunNetwork: failed to open SRAM capacity CSV file " + csv_path.string());
  }

  ofs << "model,layer_id,layer_name,layer_kind,"
         "isb_capacity_bytes,filter_capacity_bytes,output_spine_capacity_bytes\n";

  for (const auto& row : rows) {
    ofs << model_name << ','
        << row.layer_id << ','
        << std::quoted(row.layer_name) << ','
        << LayerKindToString(row.kind) << ','
        << row.sram_stats.input_spine_capacity_bytes << ','
        << row.sram_stats.filter_capacity_bytes << ','
        << row.sram_stats.output_spine_capacity_bytes << '\n';
  }
  ofs.flush();
  std::cout << "[Simulation] SRAM capacity CSV written to " << csv_path << "\n";
}

void WritePerLayerSramTables(const std::string& repo_name,
                             const std::string& model_name,
                             const std::vector<LayerStageRecord>& rows) {
  const auto dir_path = BuildLayerTablesDir(repo_name, model_name);
  std::filesystem::create_directories(dir_path);

  for (const auto& row : rows) {
    const std::uint64_t total_cycles =
        row.cycles.load_cycles + row.cycles.compute_cycles + row.cycles.store_cycles;
    const auto layer_file =
        dir_path / (std::string("layer_") + std::to_string(row.layer_id) + ".csv");
    std::ofstream ofs(layer_file, std::ios::out | std::ios::trunc);
    if (!ofs) {
      throw std::runtime_error("RunNetwork: failed to open per-layer SRAM CSV file " +
                               layer_file.string());
    }

    ofs << "component,access_cycles,access_cycles_over_total_cycles,capacity_bytes\n";

    auto emit_row = [&](const char* name,
                        const CoreSramStats::Component& comp,
                        std::uint64_t capacity_bytes) {
      const long double ratio = (total_cycles == 0)
                                    ? 0.0L
                                    : static_cast<long double>(comp.access_cycles) /
                                          static_cast<long double>(total_cycles);
      ofs << name << ','
          << comp.access_cycles << ','
          << std::fixed << std::setprecision(6) << ratio << ','
          << capacity_bytes << '\n';
    };

    emit_row("input_spine_buffer", row.sram_stats.input_spine,
             row.sram_stats.input_spine_capacity_bytes);
    emit_row("filter_buffer", row.sram_stats.filter,
             row.sram_stats.filter_capacity_bytes);
    emit_row("output_spine_buffer", row.sram_stats.output_spine,
             row.sram_stats.output_spine_capacity_bytes);

    ofs.flush();
  }
  std::cout << "[Simulation] Per-layer SRAM tables written to " << dir_path << "\n";
}

} // namespace

static LayerKind ParseKind_(const std::string& s) {
  if (s == "conv") return LayerKind::kConv;
  if (s == "fc")   return LayerKind::kFC;
  throw std::invalid_argument("Unknown layer kind: " + s);
}

std::vector<LayerSpec> ParseConfig(const std::string& json_path) {
  // Read entire JSON file as text.
  std::ifstream ifs(json_path);
  if (!ifs) throw std::runtime_error("ParseConfig: cannot open json file: " + json_path);
  std::string jtxt((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());

  // Parse json.
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

    s.name      = jl.value("name", std::string("L") + std::to_string(s.L));
    s.kind      = ParseKind_(jl.at("kind").get<std::string>());
    s.threshold_ = jl.value("threshold", 0.0f); // ensure float default

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

      // For now, we only support dilation = 1.
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

    // ---- Minimal quantization metadata for weights ----
    if (jl.contains("weight_q_format") && jl["weight_q_format"].is_object()) {
      const auto& qf = jl["weight_q_format"];
      s.w_bits      = qf.value("bits", 8);
      s.w_signed    = qf.value("signed", true);
      s.w_frac_bits = qf.value("frac_bits", -1);
      s.has_w_qformat = true;
    }

    // weight_scale (preferred) or legacy "weight_qparams.scale".
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

    // Optional debug provenance
    if (jl.contains("weight_float_min")) s.w_float_min = jl.at("weight_float_min").get<float>();
    if (jl.contains("weight_float_max")) s.w_float_max = jl.at("weight_float_max").get<float>();

    // Consistency checks (non-fatal warnings)
    if (s.has_w_qformat && s.has_w_scale && s.w_frac_bits >= 0) {
      const float expect = std::ldexp(1.0f, -s.w_frac_bits); // 2^-n
      const float eps = 1e-6f * std::max(1.0f, std::abs(expect));
      if (std::abs(s.w_scale - expect) > eps) {
        std::cerr << "[ParseConfig][Warn] L=" << s.L
                  << " weight_scale (" << s.w_scale
                  << ") != 2^-frac_bits (" << expect
                  << "). Proceeding with provided values.\n";
      }
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

  // Keep layers ordered by L ascending just in case.
  std::sort(out.begin(), out.end(), [](const LayerSpec& a, const LayerSpec& b){ return a.L < b.L; });
  return out;
}

sf::dram::SimpleDRAM InitDram(const std::string& bin_path, const std::string& json_path) {
  // Delegate to the convenience factory; it also builds per-layer tables.
  return sf::dram::SimpleDRAM::FromFiles(bin_path, json_path);
}

void RunNetwork(const std::vector<LayerSpec>& specs,
                sf::dram::SimpleDRAM* dram,
                const std::string& repo_name,
                const std::string& model_name) {
  if (!dram) throw std::invalid_argument("RunNetwork: null DRAM pointer");

  std::vector<LayerStageRecord> stage_rows;
  stage_rows.reserve(specs.size());

  for (const auto& s : specs) {
    switch (s.kind) {
      case LayerKind::kConv: {
        ConvLayer conv;
        conv.ConfigureLayer(s.L,
                            s.Cin_in, s.Cout,
                            s.H_in,   s.W_in,
                            s.Kh,     s.Kw,
                            s.Sh,     s.Sw,
                            s.Ph,     s.Pw,
                            s.threshold_,
                            s.w_bits,
                            s.w_signed,
                            s.w_frac_bits,
                            s.w_scale,
                            dram);
        conv.run_layer();
        stage_rows.push_back(LayerStageRecord{
            s.L,
            s.name,
            s.kind,
            conv.cycle_stats(),
            conv.sram_stats()
        });
        break;
      }
      case LayerKind::kFC: {
        FCLayer fc;
        fc.ConfigureLayer(s.L,
                          s.Cin_in, s.Cout,
                          s.H_in,   s.W_in,
                          s.Kh,     s.Kw,
                          s.Sh,     s.Sw,
                          s.Ph,     s.Pw,
                          s.threshold_,
                          s.w_bits,
                          s.w_signed,
                          s.w_frac_bits,
                          s.w_scale,
                          dram);
        fc.run_layer();
        stage_rows.push_back(LayerStageRecord{
            s.L,
            s.name,
            s.kind,
            fc.cycle_stats(),
            fc.sram_stats()
        });
        break;
      }
      default:
        throw std::runtime_error("RunNetwork: unsupported layer kind at L=" + std::to_string(s.L));
    }
  }

  WriteStageCyclesCsv(repo_name, model_name, stage_rows);
  WriteSramAccessCsv(repo_name, model_name, stage_rows);
  WriteSramCapacityCsv(repo_name, model_name, stage_rows);
  WritePerLayerSramTables(repo_name, model_name, stage_rows);
}

} // namespace sf
