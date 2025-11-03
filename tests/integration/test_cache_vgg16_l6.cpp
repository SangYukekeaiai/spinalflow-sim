// All comments are in English.
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "cache/cache_config.h"
#include "cache/cache_iface.h"
#include "common/constants.hpp"
#include "model/conv_layer.hpp"
#include "model/fc_layer.hpp"
#include "runner/simulation.hpp"
#include "stats/layer_stats_csv.h"
#include "stats/reuse_distance_tracker.h"
#include "stats/reuse_in_tile_tracker.h"
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
  const std::filesystem::path cross_csv =
      cross_stats_root / "288KB_32ways_lru.csv";
  test::stats::EnsureCsvHasHeader(cross_csv);

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

      auto dram = sf::InitDram(bin_path.string(), json_path.string());

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
      cache_cfg.Validate();

      auto report_stats = [&](const sf::cache::CacheStats& stats) {
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
            test::stats::LayerStatsSummary::FromCacheStats(spec.L, stats, cache_cfg);
        test::stats::AppendRow(cross_csv, summary);
      };

      bool reuse_written = false;
      const int tiles_per_spine = std::max(
          1,
          std::min(static_cast<int>(sf::kTilesPerSpine),
                   (spec.Cout + static_cast<int>(sf::kNumPE) - 1) /
                       static_cast<int>(sf::kNumPE)));

      if (reuse_in_tile_enabled) {
        reuse_tile_tracker.BeginLayer(spec.L, tiles_per_spine);
      }

      if (spec.kind == sf::LayerKind::kConv) {
        sf::ConvLayer layer;
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
                             &dram);

        layer.OverrideWeightCache(cache_cfg);

        if (reuse_hist_enabled) {
          reuse_tracker.BeginLayer(spec.L);
        }
        if (spike_stats_enabled) {
          spike_tracker.BeginLayer(spec.L, tiles_per_spine);
        }

        layer.run_layer();

        const auto* stats = layer.weight_cache_stats();
        if (!stats) {
          std::cerr << "Weight cache stats unavailable." << std::endl;
          return 2;
        }

        report_stats(*stats);
        reuse_written = true;
      } else {
        sf::FCLayer layer;
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
                             &dram);

        layer.OverrideWeightCache(cache_cfg);

        if (reuse_hist_enabled) {
          reuse_tracker.BeginLayer(spec.L);
        }
        if (spike_stats_enabled) {
          spike_tracker.BeginLayer(spec.L, tiles_per_spine);
        }

        layer.run_layer();

        const auto* stats = layer.weight_cache_stats();
        if (!stats) {
          std::cerr << "Weight cache stats unavailable." << std::endl;
          return 2;
        }

        report_stats(*stats);
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
