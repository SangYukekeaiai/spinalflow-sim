// All comments are in English.
#include "utils/stats_io.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <stdexcept>

namespace sf {

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

void WritePerSetUniqueDemandLinesCsv(const std::filesystem::path& csv_path,
                                     int num_sets,
                                     const std::unordered_map<int, std::uint64_t>& counts) {
  std::filesystem::create_directories(csv_path.parent_path());
  std::ofstream ofs(csv_path, std::ios::out | std::ios::trunc);
  if (!ofs) {
    throw std::runtime_error("RunNetwork: failed to open per-set unique demand lines CSV " + csv_path.string());
  }
  ofs << "set_idx,unique_demand_lines\n";
  (void)num_sets; // sets with zero counts are omitted by request
  std::vector<std::pair<int, std::uint64_t>> entries;
  entries.reserve(counts.size());
  for (const auto& kv : counts) {
    if (kv.second > 0ULL) entries.emplace_back(kv.first, kv.second);
  }
  std::sort(entries.begin(), entries.end(),
            [](const auto& a, const auto& b){ return a.first < b.first; });
  for (const auto& [set_idx, c] : entries) {
    ofs << set_idx << ',' << c << '\n';
  }
  ofs.flush();
}

std::filesystem::path BuildSramAccessCsvPath(const std::string& repo_name,
                                             const std::string& model_name) {
  const auto sanitized_repo  = SanitizeName(repo_name);
  const auto sanitized_model = SanitizeName(model_name);
  std::filesystem::path dir("stats");
  std::filesystem::path file =
      sanitized_repo + std::string("__") + sanitized_model + "__sram_access.csv";
  return dir / file;
}

std::filesystem::path BuildSramCapacityCsvPath(const std::string& repo_name,
                                               const std::string& model_name) {
  const auto sanitized_repo  = SanitizeName(repo_name);
  const auto sanitized_model = SanitizeName(model_name);
  std::filesystem::path dir("stats");
  std::filesystem::path file =
      sanitized_repo + std::string("__") + sanitized_model + "__sram_capacity.csv";
  return dir / file;
}

std::filesystem::path BuildStageCsvPath(const std::string& repo_name,
                                        const std::string& model_name) {
  const auto sanitized_repo  = SanitizeName(repo_name);
  const auto sanitized_model = SanitizeName(model_name);
  std::filesystem::path dir("stats");
  std::filesystem::path file =
      sanitized_repo + std::string("__") + sanitized_model + "__stage_cycles.csv";
  return dir / file;
}

std::filesystem::path BuildLayerTablesDir(const std::string& repo_name,
                                          const std::string& model_name) {
  const auto sanitized_repo  = SanitizeName(repo_name);
  const auto sanitized_model = SanitizeName(model_name);
  std::filesystem::path dir("stats");
  dir /= sanitized_repo + std::string("__") + sanitized_model;
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

static std::string BuildCacheConfigCsvName(std::size_t cache_size_kb,
                                           int cache_ways,
                                           int prefetch_depth,
                                           const std::string& policy_tag) {
  return std::to_string(cache_size_kb) + "KB_" +
         std::to_string(cache_ways) + "_" +
         std::to_string(prefetch_depth) + "_" +
         policy_tag + ".csv";
}

void WriteCacheConfigCsvs(const std::string& repo_name,
                          const std::string& model_name,
                          const std::string& policy_tag,
                          int cache_ways,
                          int prefetch_depth,
                          std::size_t cache_size_kb,
                          const std::vector<LayerStageRecord>& stage_rows,
                          bool write_stats_csv,
                          bool write_reuse_distribution_csv,
                          bool /*write_visit_count_distribution_csv*/,
                          bool write_per_set_unique_csv,
                          bool single_layer_run,
                          bool is_lru_policy,
                          std::vector<std::pair<int, CacheTotalsRow>>* per_layer_rows_out,
                          CacheTotalsRow* model_row_out) {
  const std::filesystem::path csv_dir = std::filesystem::path("stats") / repo_name / model_name;
  std::filesystem::create_directories(csv_dir);

  const std::string csv_name = BuildCacheConfigCsvName(cache_size_kb,
                                                       cache_ways,
                                                       prefetch_depth,
                                                       policy_tag);

  std::ofstream ofs;
  if (write_stats_csv && !single_layer_run) {
    const auto csv_path = csv_dir / csv_name;
    ofs.open(csv_path, std::ios::out | std::ios::trunc);
    if (!ofs) {
      throw std::runtime_error("RunNetwork: failed to open cache summary CSV " + csv_path.string());
    }
    ofs << "layer,demand_accesses,hits,misses,hit_cycles,miss_cycles,total_cycles,hit_rate,"
           "prefetch_requests,unique_demand_lines,avg_weight_reuse,avg_reuse_distance";
    if (!is_lru_policy) {
      ofs << ",used_prefetches,prefetch_use_rate";
    }
    ofs << '\n';
  }

  sf::arch::cache::CacheStats layer_totals{};

  for (const auto& row : stage_rows) {
    const auto& cs = row.cache_stats;
    const std::uint64_t hits = (cs.demand_accesses >= cs.demand_misses)
                                   ? (cs.demand_accesses - cs.demand_misses)
                                   : 0ULL;
    const std::uint64_t total_cycles_layer = cs.demand_hit_cycles + cs.demand_miss_cycles;
    double layer_hit_rate = 0.0;
    if (cs.demand_accesses > 0) {
      layer_hit_rate = static_cast<double>(hits) /
                       static_cast<double>(cs.demand_accesses);
    }

    const double layer_avg_reuse = (cs.unique_demand_lines > 0)
                                       ? static_cast<double>(cs.demand_accesses) /
                                             static_cast<double>(cs.unique_demand_lines)
                                       : 0.0;
    const double layer_avg_reuse_distance = (cs.reuse_events > 0)
                                                ? static_cast<double>(cs.reuse_distance_total) /
                                                      static_cast<double>(cs.reuse_events)
                                                : 0.0;

    const std::uint64_t total_prefetch_slots =
        static_cast<std::uint64_t>(cs.prefetch_requests) *
        static_cast<std::uint64_t>(prefetch_depth);
    const std::uint64_t layer_used_prefetches =
        (total_prefetch_slots >= cs.zero_score_events)
            ? (total_prefetch_slots - cs.zero_score_events)
            : 0ULL;
    const double layer_prefetch_use_rate =
        (total_prefetch_slots > 0)
            ? static_cast<double>(layer_used_prefetches) /
                  static_cast<double>(total_prefetch_slots)
            : 0.0;

    if (write_stats_csv && !single_layer_run) {
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
    }

    // Accumulate into totals
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

    // Per-layer CSV and reuse distribution
    const std::filesystem::path layer_dir =
        csv_dir / (std::string("layer") + std::to_string(row.layer_id));
    std::filesystem::create_directories(layer_dir);
    if (write_stats_csv) {
      const auto per_layer_csv_path = layer_dir / csv_name;
      std::ofstream lfs(per_layer_csv_path, std::ios::out | std::ios::trunc);
      if (!lfs) {
        throw std::runtime_error("RunNetwork: failed to open per-layer cache CSV " + per_layer_csv_path.string());
      }
      lfs << "layer,demand_accesses,hits,misses,hit_cycles,miss_cycles,total_cycles,hit_rate,"
             "prefetch_requests,unique_demand_lines,avg_weight_reuse,avg_reuse_distance";
      if (!is_lru_policy) {
        lfs << ",used_prefetches,prefetch_use_rate";
      }
      lfs << '\n';
      // Single row
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
      // Duplicate as total row
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
    }

    if (write_reuse_distribution_csv) {
      const auto per_layer_reuse_csv_path = layer_dir /
          (std::string("reuse_distribution_") +
           std::to_string(cache_size_kb) + "KB_" +
           std::to_string(cache_ways) + "_" +
           std::to_string(prefetch_depth) + "_" +
           policy_tag + ".csv");
      WriteReuseDistributionCsv(per_layer_reuse_csv_path,
                                cs.reuse_distance_histogram);
    }

    if (write_per_set_unique_csv) {
      // Emit per-set unique demand line counts: omit zero rows
      // Compute number of sets from cache parameters (assumes 128B line size)
      constexpr std::size_t kLineBytes = 128;
      const std::size_t total_lines = (cache_size_kb * 1024ULL) / kLineBytes;
      const int num_sets = static_cast<int>(total_lines / static_cast<std::size_t>(cache_ways));
      const auto per_layer_set_csv_path = layer_dir /
          (std::string("set_unique_demand_lines_") +
           std::to_string(cache_size_kb) + "KB_" +
           std::to_string(cache_ways) + "_" +
           std::to_string(prefetch_depth) + "_" +
           policy_tag + ".csv");
      WritePerSetUniqueDemandLinesCsv(per_layer_set_csv_path,
                                      (num_sets > 0 ? num_sets : 1),
                                      cs.per_set_unique_demand_lines);
    }

    // Prepare per-layer totals row
    if (per_layer_rows_out) {
      CacheTotalsRow pl;
      pl.cache_size_kb        = cache_size_kb;
      pl.demand_accesses      = cs.demand_accesses;
      pl.hits                 = hits;
      pl.misses               = cs.demand_misses;
      pl.hit_cycles           = cs.demand_hit_cycles;
      pl.miss_cycles          = cs.demand_miss_cycles;
      pl.total_cycles         = total_cycles_layer;
      pl.hit_rate             = layer_hit_rate;
      pl.prefetch_requests    = cs.prefetch_requests;
      pl.unique_demand_lines  = cs.unique_demand_lines;
      pl.avg_weight_reuse     = layer_avg_reuse;
      pl.zero_score_events    = cs.zero_score_events;
      pl.used_prefetches      = layer_used_prefetches;
      pl.prefetch_use_rate    = layer_prefetch_use_rate;
      pl.reuse_distance_total = cs.reuse_distance_total;
      pl.reuse_events         = cs.reuse_events;
      pl.avg_reuse_distance   = layer_avg_reuse_distance;
      per_layer_rows_out->emplace_back(row.layer_id, pl);
    }
  }

  // Do not write a model-level reuse distribution CSV to avoid duplicating
  // reuse_distribution_*.csv files at both the model directory and per-layer
  // directories. Only per-layer reuse distributions are emitted above when
  // write_reuse_distribution_csv is true.

  // Model-level total row (skip in single-layer runs to avoid duplicates)
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
      static_cast<std::uint64_t>(prefetch_depth);
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

  if (write_stats_csv && !single_layer_run) {
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
  }

  if (model_row_out) {
    *model_row_out = CacheTotalsRow{
        cache_size_kb,
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
    };
  }
}

void WriteAggregatedCacheTotalsCsvs(const std::string& repo_name,
                                    const std::string& model_name,
                                    const std::string& policy_tag,
                                    int cache_ways,
                                    int prefetch_depth,
                                    bool is_lru_policy,
                                    bool write_stats_csv,
                                    bool single_layer_run,
                                    const std::vector<CacheTotalsRow>& cache_total_rows,
                                    const std::unordered_map<int, std::vector<CacheTotalsRow>>& per_layer_totals_rows) {
  if (!write_stats_csv) return;
  // Model-level aggregated totals: write only when not a single-layer run
  if (!single_layer_run) {
    const std::filesystem::path summary_dir = std::filesystem::path("stats") / repo_name / model_name;
    std::filesystem::create_directories(summary_dir);
    const auto summary_path = summary_dir /
        ("cache_totals_" + std::to_string(cache_ways) + "ways_" +
         std::to_string(prefetch_depth) + "prefetchs_" +
         policy_tag + ".csv");
    std::ofstream totals_ofs(summary_path, std::ios::out | std::ios::trunc);
    if (!totals_ofs) {
      throw std::runtime_error("RunNetwork: failed to open aggregated cache totals CSV " + summary_path.string());
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

  // Per-layer aggregated totals
  for (const auto& kv : per_layer_totals_rows) {
    const int layer_id = kv.first;
    const auto& rows = kv.second;
    const std::filesystem::path layer_dir =
        std::filesystem::path("stats") / repo_name / model_name /
        (std::string("layer") + std::to_string(layer_id));
    std::filesystem::create_directories(layer_dir);
    const auto layer_summary_path = layer_dir /
        ("cache_totals_" + std::to_string(cache_ways) + "ways_" +
         std::to_string(prefetch_depth) + "prefetchs_" +
         policy_tag + ".csv");
    std::ofstream ltotals_ofs(layer_summary_path, std::ios::out | std::ios::trunc);
    if (!ltotals_ofs) {
      throw std::runtime_error("RunNetwork: failed to open per-layer cache totals CSV " + layer_summary_path.string());
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

} // namespace sf
