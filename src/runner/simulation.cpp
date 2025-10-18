// All comments are in English.
#include "runner/simulation.hpp"
#include "arch/cache/cache.hpp"
#include <fstream>
#include <iterator>
#include <algorithm>
#include <cmath>     // for std::ldexp, std::abs
#include <filesystem>
#include <cctype>
#include <iomanip>
#include <unordered_map>

using nlohmann::json;

namespace sf {

namespace {

struct LayerStageRecord {
  int layer_id = 0;
  std::string layer_name;
  LayerKind kind = LayerKind::kConv;
  CoreCycleStats cycles{};
  CoreSramStats sram_stats{};
  sf::arch::cache::CacheStats cache_stats{};
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

const char* EvictionPolicyToString(sf::arch::cache::EvictionPolicy policy) {
  using sf::arch::cache::EvictionPolicy;
  switch (policy) {
    case EvictionPolicy::kScoreboard: return "scoreboard";
    case EvictionPolicy::kLRU:        return "lru";
    default:                          return "unknown";
  }
}

void WriteReuseDistributionCsv(const std::filesystem::path& csv_path,
                               const std::unordered_map<std::uint64_t, std::uint64_t>& histogram) {
  std::filesystem::create_directories(csv_path.parent_path());
  std::ofstream ofs(csv_path, std::ios::out | std::ios::trunc);
  if (!ofs) {
    throw std::runtime_error("RunNetwork: failed to open reuse distribution CSV " + csv_path.string());
  }

  ofs << "reuse_distance,count,share\n";
  std::uint64_t total_events = 0;
  for (const auto& entry : histogram) {
    total_events += entry.second;
  }
  if (total_events == 0) {
    return;
  }

  std::vector<std::pair<std::uint64_t, std::uint64_t>> entries(histogram.begin(), histogram.end());
  std::sort(entries.begin(), entries.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

  const auto old_precision = ofs.precision();
  const auto old_flags = ofs.flags();
  ofs << std::fixed << std::setprecision(6);
  for (const auto& [distance, count] : entries) {
    const double share = static_cast<double>(count) /
                         static_cast<double>(total_events);
    ofs << distance << ',' << count << ',' << share << '\n';
  }
  ofs.flags(old_flags);
  ofs.precision(old_precision);
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

// Captures how often each SRAM-backed structure is touched; this feeds the
// SRAM access summary CSV consumed by notebooks and plotting utilities.
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

// Records the configured SRAM footprint for each component so capacity checks
// can be correlated with access intensity in post-processing.
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

void RunNetworkWithCacheOptions(const std::vector<LayerSpec>& specs,
                                sf::dram::SimpleDRAM* dram,
                                const std::string& repo_name,
                                const std::string& model_name,
                                const std::vector<std::size_t>& cache_sizes_bytes,
                                const std::vector<int>& cache_way_options,
                                const std::vector<int>& prefetch_depth_options,
                                const std::vector<sf::arch::cache::EvictionPolicy>& policies) {
  if (!dram) throw std::invalid_argument("RunNetwork: null DRAM pointer");
  if (cache_sizes_bytes.empty()) {
    throw std::invalid_argument("RunNetwork: cache_sizes_bytes is empty");
  }
  if (cache_way_options.empty()) {
    throw std::invalid_argument("RunNetwork: cache_way_options is empty");
  }
  if (prefetch_depth_options.empty()) {
    throw std::invalid_argument("RunNetwork: prefetch_depth_options is empty");
  }
  if (policies.empty()) {
    throw std::invalid_argument("RunNetwork: policies list is empty");
  }

  struct CacheTotalRow {
    std::size_t   cache_size_kb        = 0;
    std::uint64_t demand_accesses      = 0;
    std::uint64_t hits                 = 0;
    std::uint64_t misses               = 0;
    std::uint64_t hit_cycles           = 0;
    std::uint64_t miss_cycles          = 0;
    std::uint64_t total_cycles         = 0;
    double        hit_rate             = 0.0;
    std::uint64_t prefetch_requests    = 0;
    std::uint64_t unique_demand_lines  = 0;
    double        avg_weight_reuse     = 0.0;
    std::uint64_t zero_score_events    = 0;
    std::uint64_t used_prefetches      = 0;
    double        prefetch_use_rate    = 0.0;
    std::uint64_t reuse_distance_total = 0;
    std::uint64_t reuse_events         = 0;
    double        avg_reuse_distance   = 0.0;
  };
  for (const auto policy : policies) {
    const std::string policy_tag = SanitizeName(EvictionPolicyToString(policy));
    const bool is_lru_policy =
        (policy == sf::arch::cache::EvictionPolicy::kLRU);

    for (int cache_ways : cache_way_options) {
      for (int prefetch_depth : prefetch_depth_options) {
        std::vector<LayerStageRecord> final_stage_rows;
        std::vector<CacheTotalRow> cache_total_rows;
        cache_total_rows.reserve(cache_sizes_bytes.size());
        // Accumulate per-layer totals across cache sizes so we can emit a
        // cache_totals_*.csv inside each layer<N> directory.
        std::unordered_map<int, std::vector<CacheTotalRow>> per_layer_totals_rows;

        for (std::size_t cfg_idx = 0; cfg_idx < cache_sizes_bytes.size(); ++cfg_idx) {
          sf::arch::cache::CacheConfig cache_cfg{};
          cache_cfg.capacity_bytes = cache_sizes_bytes[cfg_idx];
          cache_cfg.ways = cache_ways;
          cache_cfg.prefetch_depth = prefetch_depth;
          cache_cfg.eviction_policy = policy;
          const std::size_t cache_size_kb_int = cache_cfg.capacity_bytes / 1024u;
          const std::filesystem::path stats_dir =
              std::filesystem::path("stats") / repo_name / model_name;
          std::filesystem::create_directories(stats_dir);
          const std::filesystem::path trace_dir =
              stats_dir / "cache_traces" / policy_tag /
              (std::to_string(cache_ways) + "_" + std::to_string(prefetch_depth));
          std::filesystem::create_directories(trace_dir);
          cache_cfg.trace_output_path = (trace_dir / (std::to_string(cache_size_kb_int) + ".txt")).string();
          cache_cfg.trace_max_lines = 5000;

          sf::arch::cache::CacheSim shared_cache(cache_cfg);

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
                                    dram,
                                    &shared_cache);
                conv.run_layer();
                stage_rows.push_back(LayerStageRecord{
                    s.L,
                    s.name,
                    s.kind,
                    conv.cycle_stats(),
                    conv.sram_stats(),
                    conv.cache_stats()
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
                                  dram,
                                  &shared_cache);
                fc.run_layer();
                stage_rows.push_back(LayerStageRecord{
                    s.L,
                    s.name,
                    s.kind,
                    fc.cycle_stats(),
                    fc.sram_stats(),
                    fc.cache_stats()
                });
                break;
              }
              default:
                throw std::runtime_error("RunNetwork: unsupported layer kind at L=" + std::to_string(s.L));
            }
          }

          const std::filesystem::path csv_dir = stats_dir;
          const std::string csv_name =
              std::to_string(cache_size_kb_int) + "KB_" +
              std::to_string(cache_cfg.ways) + "_" +
              std::to_string(cache_cfg.prefetch_depth) + "_" +
              policy_tag + ".csv";
          // Continue writing the aggregate file at model root for compatibility.
          const auto csv_path = csv_dir / csv_name;
          std::ofstream ofs(csv_path, std::ios::out | std::ios::trunc);
          if (!ofs) {
            throw std::runtime_error("RunNetwork: failed to open cache summary CSV " + csv_path.string());
          }
          ofs << "layer,demand_accesses,hits,misses,hit_cycles,miss_cycles,total_cycles,hit_rate,"
                 "prefetch_requests,unique_demand_lines,avg_weight_reuse,avg_reuse_distance";
          if (!is_lru_policy) {
            ofs << ",used_prefetches,prefetch_use_rate";
          }
          ofs << '\n';

          sf::arch::cache::CacheStats layer_totals{};

          for (const auto& row : stage_rows) {
            const auto& cs = row.cache_stats;
            const std::uint64_t hits =
                (cs.demand_accesses >= cs.demand_misses)
                    ? (cs.demand_accesses - cs.demand_misses)
                    : 0ULL;
            const std::uint64_t total_cycles_layer =
                cs.demand_hit_cycles + cs.demand_miss_cycles;
            double layer_hit_rate = 0.0;
            if (cs.demand_accesses > 0) {
              layer_hit_rate = static_cast<double>(hits) /
                               static_cast<double>(cs.demand_accesses);
            }

            const double layer_avg_reuse =
                (cs.unique_demand_lines > 0)
                    ? static_cast<double>(cs.demand_accesses) /
                          static_cast<double>(cs.unique_demand_lines)
                    : 0.0;
            const double layer_avg_reuse_distance =
                (cs.reuse_events > 0)
                    ? static_cast<double>(cs.reuse_distance_total) /
                          static_cast<double>(cs.reuse_events)
                    : 0.0;
            const std::uint64_t total_prefetch_slots =
                static_cast<std::uint64_t>(cs.prefetch_requests) *
                static_cast<std::uint64_t>(cache_cfg.prefetch_depth);
            const std::uint64_t layer_used_prefetches =
                (total_prefetch_slots >= cs.zero_score_events)
                    ? (total_prefetch_slots - cs.zero_score_events)
                    : 0ULL;
            const double layer_prefetch_use_rate =
                (total_prefetch_slots > 0)
                    ? static_cast<double>(layer_used_prefetches) /
                          static_cast<double>(total_prefetch_slots)
                    : 0.0;

            ofs << row.layer_id << ','
                << cs.demand_accesses << ','
                << hits << ','
                << cs.demand_misses << ','
                << cs.demand_hit_cycles << ','
                << cs.demand_miss_cycles << ','
                << total_cycles_layer << ','
                << layer_hit_rate << ','
                << cs.prefetch_requests << ','
                << cs.unique_demand_lines << ','
                << layer_avg_reuse << ','
                << layer_avg_reuse_distance;
            if (!is_lru_policy) {
              ofs << ',' << layer_used_prefetches << ','
                  << layer_prefetch_use_rate;
            }
            ofs << '\n';

            layer_totals.demand_accesses += cs.demand_accesses;
            layer_totals.demand_misses += cs.demand_misses;
            layer_totals.demand_hit_cycles += cs.demand_hit_cycles;
            layer_totals.demand_miss_cycles += cs.demand_miss_cycles;
            layer_totals.prefetch_requests += cs.prefetch_requests;
            layer_totals.prefetch_misses += cs.prefetch_misses;
            layer_totals.unique_demand_lines += cs.unique_demand_lines;
            layer_totals.zero_score_events += cs.zero_score_events;
            layer_totals.reuse_distance_total += cs.reuse_distance_total;
            layer_totals.reuse_events += cs.reuse_events;
            for (const auto& [distance, count] : cs.reuse_distance_histogram) {
              layer_totals.reuse_distance_histogram[distance] += count;
            }

            // Also emit per-layer CSV and reuse distribution into
            // stats/<repo>/<model>/layer<id>/
            const std::filesystem::path layer_dir =
                stats_dir / (std::string("layer") + std::to_string(row.layer_id));
            std::filesystem::create_directories(layer_dir);

            // Per-layer summary for this cache configuration.
            const auto per_layer_csv_path = layer_dir / csv_name;
            std::ofstream lfs(per_layer_csv_path, std::ios::out | std::ios::trunc);
            if (!lfs) {
              throw std::runtime_error(
                  "RunNetwork: failed to open per-layer cache CSV " + per_layer_csv_path.string());
            }
            lfs << "layer,demand_accesses,hits,misses,hit_cycles,miss_cycles,total_cycles,hit_rate,"
                   "prefetch_requests,unique_demand_lines,avg_weight_reuse,avg_reuse_distance";
            if (!is_lru_policy) {
              lfs << ",used_prefetches,prefetch_use_rate";
            }
            lfs << '\n';
            // Single layer row
            lfs << row.layer_id << ','
                << cs.demand_accesses << ','
                << hits << ','
                << cs.demand_misses << ','
                << cs.demand_hit_cycles << ','
                << cs.demand_miss_cycles << ','
                << total_cycles_layer << ','
                << layer_hit_rate << ','
                << cs.prefetch_requests << ','
                << cs.unique_demand_lines << ','
                << layer_avg_reuse << ','
                << layer_avg_reuse_distance;
            if (!is_lru_policy) {
              lfs << ',' << layer_used_prefetches << ','
                  << layer_prefetch_use_rate;
            }
            lfs << '\n';
            // Duplicate as total row for convenience
            lfs << "total,"
                << cs.demand_accesses << ','
                << hits << ','
                << cs.demand_misses << ','
                << cs.demand_hit_cycles << ','
                << cs.demand_miss_cycles << ','
                << total_cycles_layer << ','
                << layer_hit_rate << ','
                << cs.prefetch_requests << ','
                << cs.unique_demand_lines << ','
                << layer_avg_reuse << ','
                << layer_avg_reuse_distance;
            if (!is_lru_policy) {
              lfs << ',' << layer_used_prefetches << ','
                  << layer_prefetch_use_rate;
            }
            lfs << '\n';
            lfs.flush();

            // Per-layer reuse histogram CSV for this cache configuration
            const auto per_layer_reuse_csv_path = layer_dir / (std::string("reuse_distribution_") +
                                                               std::to_string(cache_size_kb_int) + "KB_" +
                                                               std::to_string(cache_cfg.ways) + "_" +
                                                               std::to_string(cache_cfg.prefetch_depth) + "_" +
                                                               policy_tag + ".csv");
            WriteReuseDistributionCsv(per_layer_reuse_csv_path,
                                      cs.reuse_distance_histogram);

            // Accumulate per-layer totals across cache sizes
            CacheTotalRow pl;
            pl.cache_size_kb       = cache_size_kb_int;
            pl.demand_accesses     = cs.demand_accesses;
            pl.hits                = hits;
            pl.misses              = cs.demand_misses;
            pl.hit_cycles          = cs.demand_hit_cycles;
            pl.miss_cycles         = cs.demand_miss_cycles;
            pl.total_cycles        = total_cycles_layer;
            pl.hit_rate            = layer_hit_rate;
            pl.prefetch_requests   = cs.prefetch_requests;
            pl.unique_demand_lines = cs.unique_demand_lines;
            pl.avg_weight_reuse    = layer_avg_reuse;
            pl.zero_score_events   = cs.zero_score_events;
            pl.used_prefetches     = layer_used_prefetches;
            pl.prefetch_use_rate   = layer_prefetch_use_rate;
            pl.reuse_distance_total = cs.reuse_distance_total;
            pl.reuse_events         = cs.reuse_events;
            pl.avg_reuse_distance   = layer_avg_reuse_distance;
            per_layer_totals_rows[row.layer_id].push_back(pl);
          }

          const std::uint64_t total_hits =
              (layer_totals.demand_accesses >= layer_totals.demand_misses)
                  ? (layer_totals.demand_accesses - layer_totals.demand_misses)
                  : 0ULL;
          const std::uint64_t total_cycles_sum =
              layer_totals.demand_hit_cycles + layer_totals.demand_miss_cycles;
          double total_hit_rate = 0.0;
          if (layer_totals.demand_accesses > 0) {
            total_hit_rate = static_cast<double>(total_hits) /
                             static_cast<double>(layer_totals.demand_accesses);
          }
          const std::uint64_t total_prefetch_slots =
              static_cast<std::uint64_t>(layer_totals.prefetch_requests) *
              static_cast<std::uint64_t>(cache_cfg.prefetch_depth);
          const std::uint64_t total_used_prefetches =
              (total_prefetch_slots >= layer_totals.zero_score_events)
                  ? (total_prefetch_slots - layer_totals.zero_score_events)
                  : 0ULL;
          const double total_prefetch_use_rate =
              (total_prefetch_slots > 0)
                  ? static_cast<double>(total_used_prefetches) /
                        static_cast<double>(total_prefetch_slots)
                  : 0.0;
          const double total_avg_weight_reuse =
              (layer_totals.unique_demand_lines > 0)
                  ? static_cast<double>(layer_totals.demand_accesses) /
                        static_cast<double>(layer_totals.unique_demand_lines)
                  : 0.0;
          const double total_avg_reuse_distance =
              (layer_totals.reuse_events > 0)
                  ? static_cast<double>(layer_totals.reuse_distance_total) /
                        static_cast<double>(layer_totals.reuse_events)
                  : 0.0;

          const auto reuse_csv_path = csv_dir / ("reuse_distribution_" +
                                                 std::to_string(cache_size_kb_int) + "KB_" +
                                                 std::to_string(cache_cfg.ways) + "_" +
                                                 std::to_string(cache_cfg.prefetch_depth) + "_" +
                                                 policy_tag + ".csv");
          WriteReuseDistributionCsv(reuse_csv_path,
                                    layer_totals.reuse_distance_histogram);

          ofs << "total,"
              << layer_totals.demand_accesses << ','
              << total_hits << ','
              << layer_totals.demand_misses << ','
              << layer_totals.demand_hit_cycles << ','
              << layer_totals.demand_miss_cycles << ','
              << total_cycles_sum << ','
              << total_hit_rate << ','
              << layer_totals.prefetch_requests << ','
              << layer_totals.unique_demand_lines << ','
              << total_avg_weight_reuse << ','
              << total_avg_reuse_distance;
          if (!is_lru_policy) {
            ofs << ',' << total_used_prefetches << ','
                << total_prefetch_use_rate;
          }
          ofs << '\n';
          ofs.flush();

          cache_total_rows.push_back(CacheTotalRow{
              cache_size_kb_int,
              layer_totals.demand_accesses,
              total_hits,
              layer_totals.demand_misses,
              layer_totals.demand_hit_cycles,
              layer_totals.demand_miss_cycles,
              total_cycles_sum,
              total_hit_rate,
              layer_totals.prefetch_requests,
              layer_totals.unique_demand_lines,
              total_avg_weight_reuse,
              layer_totals.zero_score_events,
              total_used_prefetches,
              total_prefetch_use_rate,
              layer_totals.reuse_distance_total,
              layer_totals.reuse_events,
              total_avg_reuse_distance
          });

          if (cfg_idx + 1 == cache_sizes_bytes.size()) {
            final_stage_rows = stage_rows;
          }
        }

        if (!cache_total_rows.empty()) {
          // Write model-level totals (across all layers) as before
          const std::filesystem::path summary_dir =
              std::filesystem::path("stats") / repo_name / model_name;
          std::filesystem::create_directories(summary_dir);
          const auto summary_path = summary_dir / ("cache_totals_" +
                                                   std::to_string(cache_ways) + "ways_" +
                                                   std::to_string(prefetch_depth) + "prefetchs_" +
                                                   policy_tag + ".csv");
          std::ofstream totals_ofs(summary_path, std::ios::out | std::ios::trunc);
          if (!totals_ofs) {
            throw std::runtime_error(
                "RunNetwork: failed to open aggregated cache totals CSV " + summary_path.string());
          }
          totals_ofs << "cache_size_kb,demand_accesses,hits,misses,hit_cycles,miss_cycles,"
                        "total_cycles,hit_rate,prefetch_requests,unique_demand_lines,avg_weight_reuse,avg_reuse_distance";
          if (!is_lru_policy) {
            totals_ofs << ",used_prefetches,prefetch_use_rate";
          }
          totals_ofs << '\n';
          for (const auto& row : cache_total_rows) {
            totals_ofs << row.cache_size_kb << ','
                       << row.demand_accesses << ','
                       << row.hits << ','
                       << row.misses << ','
                       << row.hit_cycles << ','
                       << row.miss_cycles << ','
                       << row.total_cycles << ','
                       << row.hit_rate << ','
                       << row.prefetch_requests << ','
                       << row.unique_demand_lines << ','
                       << row.avg_weight_reuse << ','
                       << row.avg_reuse_distance;
            if (!is_lru_policy) {
              totals_ofs << ',' << row.used_prefetches << ','
                         << row.prefetch_use_rate;
            }
            totals_ofs << '\n';
          }
          totals_ofs.flush();
        }

        // Write per-layer cache_totals_*.csv into each layer directory
        if (!per_layer_totals_rows.empty()) {
          for (const auto& kv : per_layer_totals_rows) {
            const int layer_id = kv.first;
            const auto& rows = kv.second;
            const std::filesystem::path layer_dir =
                std::filesystem::path("stats") / repo_name / model_name /
                (std::string("layer") + std::to_string(layer_id));
            std::filesystem::create_directories(layer_dir);
            const auto layer_summary_path = layer_dir / ("cache_totals_" +
                                                        std::to_string(cache_ways) + "ways_" +
                                                        std::to_string(prefetch_depth) + "prefetchs_" +
                                                        policy_tag + ".csv");
            std::ofstream ltotals_ofs(layer_summary_path, std::ios::out | std::ios::trunc);
            if (!ltotals_ofs) {
              throw std::runtime_error(
                  "RunNetwork: failed to open per-layer cache totals CSV " + layer_summary_path.string());
            }
            ltotals_ofs << "cache_size_kb,demand_accesses,hits,misses,hit_cycles,miss_cycles,"
                           "total_cycles,hit_rate,prefetch_requests,unique_demand_lines,avg_weight_reuse,avg_reuse_distance";
            if (!is_lru_policy) {
              ltotals_ofs << ",used_prefetches,prefetch_use_rate";
            }
            ltotals_ofs << '\n';
            for (const auto& row : rows) {
              ltotals_ofs << row.cache_size_kb << ','
                          << row.demand_accesses << ','
                          << row.hits << ','
                          << row.misses << ','
                          << row.hit_cycles << ','
                          << row.miss_cycles << ','
                          << row.total_cycles << ','
                          << row.hit_rate << ','
                          << row.prefetch_requests << ','
                          << row.unique_demand_lines << ','
                          << row.avg_weight_reuse << ','
                          << row.avg_reuse_distance;
              if (!is_lru_policy) {
                ltotals_ofs << ',' << row.used_prefetches << ','
                            << row.prefetch_use_rate;
              }
              ltotals_ofs << '\n';
            }
            ltotals_ofs.flush();
          }
        }

        // if (!final_stage_rows.empty()) {
        //   WriteStageCyclesCsv(repo_name, model_name, final_stage_rows);
        //   WriteSramAccessCsv(repo_name, model_name, final_stage_rows);
        //   WriteSramCapacityCsv(repo_name, model_name, final_stage_rows);
        //   WritePerLayerSramTables(repo_name, model_name, final_stage_rows);
        // }
      }
    }
  }
}

void RunNetwork(const std::vector<LayerSpec>& specs,
                sf::dram::SimpleDRAM* dram,
                const std::string& repo_name,
                const std::string& model_name) {
  const std::vector<std::size_t> default_cache_sizes_bytes = {
      72u * 1024u,
      144u * 1024u,
      288u * 1024u,
      576u * 1024u
  };
  const std::vector<int> default_cache_way_options = {4, 8, 16};
  const std::vector<int> default_prefetch_depth_options = {1, 2, 3, 4};
  const std::vector<sf::arch::cache::EvictionPolicy> default_policies = {
      sf::arch::cache::EvictionPolicy::kScoreboard,
      sf::arch::cache::EvictionPolicy::kLRU
  };

  RunNetworkWithCacheOptions(specs,
                             dram,
                             repo_name,
                             model_name,
                             default_cache_sizes_bytes,
                             default_cache_way_options,
                             default_prefetch_depth_options,
                             default_policies);
}

} // namespace sf
