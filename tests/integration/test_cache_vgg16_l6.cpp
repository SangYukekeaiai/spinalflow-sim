// All comments are in English.
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "cache/cache_config.h"
#include "cache/cache_iface.h"
#include "cache/belady.h"
#include "common/constants.hpp"
#include "model/conv_layer.hpp"
#include "model/fc_layer.hpp"
#include "runner/simulation.hpp"
#include "stats/layer_stats_csv.h"
#include "stats/reuse_distance_tracker.h"
#include "stats/reuse_in_tile_tracker.h"
#include "stats/cache_trace_recorder.h"
#include "stats/tile_input_hooks.h"
#include "stats/tracking_cache.h"
#include "stats/spike_event_hooks.h"
#include "stats/spike_event_tracker.h"

namespace {

struct SweepPoint {
  int capacity_kb;
  int ways;
};

std::vector<SweepPoint> BuildSweep() {
  std::vector<int> capacities = {288};
  std::vector<int> ways = {32};
  std::vector<SweepPoint> sweep;
  sweep.reserve(capacities.size() * ways.size());
  for (int cap : capacities) {
    for (int way : ways) {
      sweep.push_back({cap, way});
    }
  }
  return sweep;
}

} // namespace

int main(int argc, char** argv) {
  const std::string repo = "repo4";
  const std::string model = "vgg16";
  const std::filesystem::path repo_root = std::filesystem::current_path().parent_path();
  const std::filesystem::path base = repo_root / "workloads" / repo / model;
  const std::filesystem::path json_path = base / "dram_meta.json";
  const std::filesystem::path bin_path  = base / "dram_image.bin";

  auto specs = sf::ParseConfig(json_path.string());
  std::set<int> requested_layers;
  bool reuse_hist_enabled = false;
  bool spike_stats_enabled = false;
  bool reuse_in_tile_enabled = false;
  bool belady_enabled = false;
  bool prefetch_buffer_enabled = true;
  std::optional<int> prefetch_buffer_lines_cli;
  std::optional<int> prefetch_buffer_kb_cli;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--reuse-dist") {
      reuse_hist_enabled = true;
      continue;
    }
    if (arg == "--spike-stats") {
      spike_stats_enabled = true;
      continue;
    }
    if (arg == "--reuse-in-tile") {
      reuse_in_tile_enabled = true;
      continue;
    }
    if (arg == "--belady") {
      belady_enabled = true;
      continue;
    }
    if (arg == "--no-prefetch-buffer") {
      prefetch_buffer_enabled = false;
      continue;
    }
    if (arg == "--prefetch-buffer") {
      prefetch_buffer_enabled = true;
      continue;
    }
    const std::string lines_prefix = "--prefetch-buffer-lines=";
    const std::string kb_prefix = "--prefetch-buffer-kb=";
    if (arg.rfind(lines_prefix, 0) == 0) {
      prefetch_buffer_enabled = true;
      prefetch_buffer_lines_cli = std::stoi(arg.substr(lines_prefix.size()));
      continue;
    }
    if (arg.rfind(kb_prefix, 0) == 0) {
      prefetch_buffer_enabled = true;
      prefetch_buffer_kb_cli = std::stoi(arg.substr(kb_prefix.size()));
      continue;
    }
    requested_layers.insert(std::atoi(argv[i]));
  }

  const auto sweep = BuildSweep();

  std::cout << "layer,capacity_kb,ways,num_sets,latency_cycles,demand_hits,demand_miss_alloc,demand_miss_noalloc,prefetch_hits,hit_rate\n";

  const std::filesystem::path csv_root = std::filesystem::path("stats") / repo / model;
  std::filesystem::create_directories(csv_root);
  const std::filesystem::path csv_path = csv_root / "cache_stats.csv";
  const bool csv_exists = std::filesystem::exists(csv_path);
  std::ofstream csv_ofs(csv_path, std::ios::out | std::ios::app);
  if (!csv_ofs) {
    std::cerr << "Failed to open " << csv_path << " for writing.\n";
    return 3;
  }
  if (!csv_exists) {
    csv_ofs << "layer,capacity_kb,ways,num_sets,latency_cycles,demand_hits,demand_miss_alloc,demand_miss_noalloc,prefetch_hits,hit_rate\n";
  }

  const std::filesystem::path cross_stats_root =
      csv_root / "cross_layer_comparsion_stats";
  std::filesystem::create_directories(cross_stats_root);

  test::stats::ReuseDistanceTracker reuse_tracker;
  if (reuse_hist_enabled) {
    test::stats::EnableReuseTracking(reuse_tracker);
  }
  test::stats::SpikeEventTracker spike_tracker;
  if (spike_stats_enabled) {
    test::stats::EnableSpikeEventTracking(spike_tracker);
  }
  test::stats::ReuseInTileTracker reuse_tile_tracker;
  if (reuse_in_tile_enabled) {
    test::stats::EnableTileReuseTracking(reuse_tile_tracker);
  }
  const std::filesystem::path spike_stats_root =
      csv_root / "spiking_event_stats";
  const std::filesystem::path reuse_in_tile_root =
      csv_root / "reuse_in_tiles_statistics";

  for (const auto& spec : specs) {
    if (!requested_layers.empty() && !requested_layers.count(spec.L)) continue;
    if (spec.kind != sf::LayerKind::kConv && spec.kind != sf::LayerKind::kFC) continue;

    for (const auto& point : sweep) {
      const std::uint64_t bytes = static_cast<std::uint64_t>(point.capacity_kb) * 1024ULL;
      const std::uint64_t line_size = 128ULL;
      if (bytes % line_size != 0) {
        std::cerr << "Capacity " << point.capacity_kb << "KB is not aligned to 128B lines." << std::endl;
        continue;
      }
      const std::uint64_t total_lines = bytes / line_size;
      if (total_lines % static_cast<std::uint64_t>(point.ways) != 0) {
        std::cerr << "Capacity " << point.capacity_kb << "KB cannot be divided into "
                  << point.ways << " ways." << std::endl;
        continue;
      }
      const int num_sets = static_cast<int>(total_lines / static_cast<std::uint64_t>(point.ways));
      if ((num_sets % 2) != 0) {
        std::cerr << "num_sets must be even but was " << num_sets << " for capacity "
                  << point.capacity_kb << "KB." << std::endl;
        continue;
      }

      sf::cache::CacheConfig cache_cfg;
      cache_cfg.geometry.num_sets = num_sets;
      cache_cfg.geometry.ways = point.ways;
      cache_cfg.geometry.line_size_bytes = static_cast<int>(line_size);
      cache_cfg.timing.hit_latency_cycles = 1;
      cache_cfg.timing.miss_latency_cycles = 40;
      cache_cfg.replacement_kind = sf::cache::ReplacementKind::Lru;
      cache_cfg.Cin = spec.Cin_in;
      cache_cfg.KH = spec.Kh;
      cache_cfg.KW = spec.Kw;
      cache_cfg.prefetch_buffer_enabled = prefetch_buffer_enabled;
      if (cache_cfg.prefetch_buffer_enabled) {
        int buffer_lines = cache_cfg.prefetch_buffer_capacity_lines;
        if (prefetch_buffer_lines_cli) {
          buffer_lines = *prefetch_buffer_lines_cli;
        } else if (prefetch_buffer_kb_cli) {
          const std::uint64_t requested_bytes =
              static_cast<std::uint64_t>(*prefetch_buffer_kb_cli) * 1024ULL;
          if (requested_bytes % line_size != 0) {
            std::cerr << "Prefetch buffer size " << *prefetch_buffer_kb_cli
                      << "KB is not compatible with line size " << line_size << "B.\n";
            continue;
          }
          buffer_lines = static_cast<int>(requested_bytes / line_size);
        }
        if (buffer_lines <= 0) {
          std::cerr << "Prefetch buffer capacity must be positive.\n";
          continue;
        }
        cache_cfg.prefetch_buffer_capacity_lines = buffer_lines;
      } else {
        cache_cfg.prefetch_buffer_capacity_lines = 0;
      }
      if (belady_enabled) {
        cache_cfg.prefetch_buffer_enabled = false;
        cache_cfg.prefetch_buffer_capacity_lines = 0;
      }
      cache_cfg.Validate();

      const std::filesystem::path cross_csv =
          cross_stats_root /
          (std::to_string(point.capacity_kb) + "KB_" +
           std::to_string(point.ways) + "ways_" +
           ([&]() {
             if (belady_enabled) {
               return std::string("belady");
             }
             if (!cache_cfg.prefetch_buffer_enabled) {
               return std::string("lru_no_prefetch_buffer");
             }
             const std::uint64_t buffer_bytes =
                 static_cast<std::uint64_t>(cache_cfg.prefetch_buffer_capacity_lines) *
                 static_cast<std::uint64_t>(cache_cfg.geometry.line_size_bytes);
             std::ostringstream oss;
             oss << "lru_prefetch_buffer_";
             if (buffer_bytes % 1024ULL == 0) {
               oss << (buffer_bytes / 1024ULL) << "KB";
             } else {
               oss << buffer_bytes << "B";
             }
             return oss.str();
           })() +
           ".csv");
      test::stats::EnsureCsvHasHeader(cross_csv);

      auto report_stats = [&](const sf::cache::CacheStats& stats,
                              const sf::cache::CacheConfig& effective_cfg) {
        const std::uint64_t total_demand =
            stats.demand_hits +
            stats.demand_misses_allocated +
            stats.demand_misses_noalloc;
        const double hit_rate =
            (total_demand == 0)
                ? 0.0
                : static_cast<double>(stats.demand_hits) /
                      static_cast<double>(total_demand);

        std::cout << spec.L << ','
                  << point.capacity_kb << ','
                  << point.ways << ','
                  << num_sets << ','
                  << stats.latency_cycles << ','
                  << stats.demand_hits << ','
                  << stats.demand_misses_allocated << ','
                  << stats.demand_misses_noalloc << ','
                  << stats.prefetch_hits << ','
                  << hit_rate << '\n';

        csv_ofs << spec.L << ','
                << point.capacity_kb << ','
                << point.ways << ','
                << num_sets << ','
                << stats.latency_cycles << ','
                << stats.demand_hits << ','
                << stats.demand_misses_allocated << ','
                << stats.demand_misses_noalloc << ','
                << stats.prefetch_hits << ','
                << hit_rate << '\n';

        const auto summary =
            test::stats::LayerStatsSummary::FromCacheStats(spec.L, stats, effective_cfg);
        test::stats::AppendRow(cross_csv, summary);
      };

      bool reuse_written = false;
      const int tiles_per_spine = std::max(
          1,
          std::min(static_cast<int>(sf::kTilesPerSpine),
                   (spec.Cout + static_cast<int>(sf::kNumPE) - 1) /
                       static_cast<int>(sf::kNumPE)));

      auto run_layer = [&](const sf::cache::CacheConfig& cfg,
                           bool trackers_active,
                           std::vector<sf::cache::AccessRequest>* trace)
          -> std::optional<sf::cache::CacheStats> {
        auto dram_local = sf::InitDram(bin_path.string(), json_path.string());

        auto execute = [&](auto& layer) -> std::optional<sf::cache::CacheStats> {
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
                               &dram_local);

          if (trackers_active) {
            if (reuse_hist_enabled) {
              reuse_tracker.BeginLayer(spec.L);
            }
            if (spike_stats_enabled) {
              spike_tracker.BeginLayer(spec.L, tiles_per_spine);
            }
            if (reuse_in_tile_enabled) {
              reuse_tile_tracker.BeginLayer(spec.L, tiles_per_spine);
            }
          }

          std::optional<test::stats::ScopedTraceRecorder> recorder;
          if (trace) {
            recorder.emplace(*trace);
          }

          layer.OverrideWeightCache(cfg);
          layer.run_layer();
          recorder.reset();

          const auto* stats_ptr = layer.weight_cache_stats();
          if (!stats_ptr) {
            std::cerr << "Weight cache stats unavailable." << std::endl;
            return std::nullopt;
          }
          return *stats_ptr;
        };

        if (spec.kind == sf::LayerKind::kConv) {
          sf::ConvLayer layer;
          return execute(layer);
        }
        sf::FCLayer layer;
        return execute(layer);
      };

      if (belady_enabled) {
        std::vector<sf::cache::AccessRequest> trace;
        sf::cache::CacheConfig trace_cfg = cache_cfg;
        trace_cfg.replacement_kind = sf::cache::ReplacementKind::Lru;
        trace_cfg.prefetch_buffer_enabled = false;
        trace_cfg.prefetch_buffer_capacity_lines = 0;
        auto trace_stats = run_layer(trace_cfg, false, &trace);
        if (!trace_stats.has_value()) {
          return 2;
        }

        auto plan = sf::cache::BuildBeladyPlan(cache_cfg, trace);
        sf::cache::RegisterBeladyPlan(plan);

        sf::cache::CacheConfig belady_cfg = cache_cfg;
        belady_cfg.replacement_kind = sf::cache::ReplacementKind::Belady;
        belady_cfg.prefetch_buffer_enabled = false;
        belady_cfg.prefetch_buffer_capacity_lines = 0;
        auto belady_stats = run_layer(belady_cfg, true, nullptr);
        sf::cache::ClearBeladyPlan();
        if (!belady_stats.has_value()) {
          return 2;
        }
        report_stats(*belady_stats, belady_cfg);
        reuse_written = true;
      } else {
        auto stats_opt = run_layer(cache_cfg, true, nullptr);
        if (!stats_opt.has_value()) {
          return 2;
        }
        report_stats(*stats_opt, cache_cfg);
        reuse_written = true;
      }

      if (reuse_hist_enabled && reuse_written) {
        const std::filesystem::path reuse_csv =
            csv_root / "reuse_distance_distribution" /
            ("layer" + std::to_string(spec.L) + ".csv");
        reuse_tracker.WriteCsv(reuse_csv);
      }
      if (spike_stats_enabled) {
        const std::filesystem::path spike_csv =
            spike_stats_root / ("layer_" + std::to_string(spec.L) + ".csv");
        spike_tracker.WriteCsv(spike_csv);
      }
      if (reuse_in_tile_enabled) {
        const std::filesystem::path reuse_tile_csv =
            reuse_in_tile_root / ("layer" + std::to_string(spec.L) + ".csv");
        reuse_tile_tracker.WriteCsv(reuse_tile_csv);
      }
    }
  }

  if (reuse_hist_enabled) {
    test::stats::DisableReuseTracking();
  }
  if (spike_stats_enabled) {
    test::stats::DisableSpikeEventTracking();
  }
  if (reuse_in_tile_enabled) {
    test::stats::DisableTileReuseTracking();
  }

  return 0;
}
