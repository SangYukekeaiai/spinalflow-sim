// All comments are in English.
#include "runner/simulation.hpp"
#include "utils/stats_io.hpp"
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
                                const std::vector<sf::arch::cache::EvictionPolicy>& policies,
                                bool write_stats_csv,
                                bool write_reuse_distribution_csv,
                                bool write_scoreboard_csv,
                                bool write_visit_count_distribution_csv,
                                bool write_per_set_unique_csv,
                                bool write_cache_traces) {
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

  using CacheTotalRow = sf::CacheTotalsRow;
  for (const auto policy : policies) {
    const std::string policy_tag = SanitizeName(EvictionPolicyToString(policy));
    const bool is_lru_policy = (policy == sf::arch::cache::EvictionPolicy::kLRU);
    for (std::size_t cfg_idx = 0; cfg_idx < cache_sizes_bytes.size(); ++cfg_idx) {
      for (int cache_ways : cache_way_options) {
        for (int prefetch_depth : prefetch_depth_options) {
          // Base stats directory for this model
          const std::filesystem::path stats_dir =
              std::filesystem::path("stats") / repo_name / model_name;
          std::filesystem::create_directories(stats_dir);

          // cache_size_kb_int is fixed for this configuration across layers
          const std::size_t cache_size_kb_int = cache_sizes_bytes[cfg_idx] / 1024u;

          std::vector<LayerStageRecord> stage_rows;
          stage_rows.reserve(specs.size());

          for (const auto& s : specs) {
            // Build a per-layer cache configuration so traces live under each layer
            sf::arch::cache::CacheConfig cache_cfg{};
            cache_cfg.capacity_bytes = cache_sizes_bytes[cfg_idx];
            cache_cfg.ways = cache_ways;
            cache_cfg.prefetch_depth = prefetch_depth;
            cache_cfg.eviction_policy = policy;
            cache_cfg.trace_enabled = write_cache_traces;
            if (write_cache_traces) {
              const std::filesystem::path trace_dir =
                  stats_dir / (std::string("layer") + std::to_string(s.L)) /
                  "cache_traces" / policy_tag /
                  (std::to_string(cache_ways) + "_" + std::to_string(prefetch_depth));
              std::filesystem::create_directories(trace_dir);
              cache_cfg.trace_output_path = (trace_dir / (std::to_string(cache_size_kb_int) + ".txt")).string();
              cache_cfg.trace_max_lines = 5000;
            } else {
              cache_cfg.trace_output_path.clear();
              cache_cfg.trace_max_lines = 0;
            }

            // Create a fresh cache instance per layer to isolate traces
            sf::arch::cache::CacheSim layer_cache(cache_cfg);
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
                                    &layer_cache);
                conv.run_layer();
                stage_rows.push_back(LayerStageRecord{
                    s.L,
                    s.name,
                    s.kind,
                    conv.cycle_stats(),
                    conv.sram_stats(),
                    conv.cache_stats(),
                    conv.cache_scoreboard_scores()
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
                                  &layer_cache);
                fc.run_layer();
                stage_rows.push_back(LayerStageRecord{
                    s.L,
                    s.name,
                    s.kind,
                    fc.cycle_stats(),
                    fc.sram_stats(),
                    fc.cache_stats(),
                    fc.cache_scoreboard_scores()
                });
                break;
              }
              default:
                throw std::runtime_error("RunNetwork: unsupported layer kind at L=" + std::to_string(s.L));
            }
          }

          // Write CSVs for this configuration and collect totals
          std::vector<std::pair<int, sf::CacheTotalsRow>> per_layer_rows_out;
          sf::CacheTotalsRow model_row_out;
          WriteCacheConfigCsvs(repo_name,
                               model_name,
                               policy_tag,
                               cache_ways,
                               prefetch_depth,
                               cache_size_kb_int,
                               stage_rows,
                               write_stats_csv,
                               write_reuse_distribution_csv,
                               /*write_scoreboard_csv=*/write_stats_csv, // default follow stats unless overridden upstream
                               write_visit_count_distribution_csv,
                               write_per_set_unique_csv,
                               is_lru_policy,
                               &per_layer_rows_out,
                               &model_row_out);

          // Consolidate and append totals for this single configuration
          std::unordered_map<int, std::vector<CacheTotalRow>> per_layer_totals_rows;
          for (const auto& pr : per_layer_rows_out) {
            per_layer_totals_rows[pr.first].push_back(pr.second);
          }
          std::vector<CacheTotalRow> cache_total_rows;
          cache_total_rows.push_back(CacheTotalRow{
              model_row_out.cache_size_kb,
              model_row_out.demand_accesses,
              model_row_out.hits,
              model_row_out.misses,
              model_row_out.hit_cycles,
              model_row_out.miss_cycles,
              model_row_out.total_cycles,
              model_row_out.hit_rate,
              model_row_out.prefetch_requests,
              model_row_out.unique_demand_lines,
              model_row_out.avg_weight_reuse,
              model_row_out.zero_score_events,
              model_row_out.used_prefetches,
              model_row_out.prefetch_use_rate,
              model_row_out.reuse_distance_total,
              model_row_out.reuse_events,
              model_row_out.avg_reuse_distance
          });

          WriteAggregatedCacheTotalsCsvs(repo_name,
                                         model_name,
                                         policy_tag,
                                         cache_ways,
                                         prefetch_depth,
                                         is_lru_policy,
                                         write_stats_csv,
                                         cache_total_rows,
                                         per_layer_totals_rows);
        }
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
                             default_policies,
                             /*write_stats_csv=*/true,
                             /*write_reuse_distribution_csv=*/true,
                             /*write_scoreboard_csv=*/true,
                             /*write_visit_count_distribution_csv=*/false,
                             /*write_per_set_unique_csv=*/true,
                             /*write_cache_traces=*/true);
}

} // namespace sf
