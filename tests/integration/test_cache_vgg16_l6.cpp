// All comments are in English.
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <vector>

#include "cache/cache_config.h"
#include "cache/cache_iface.h"
#include "model/conv_layer.hpp"
#include "runner/simulation.hpp"

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
  for (int i = 1; i < argc; ++i) {
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

  for (const auto& spec : specs) {
    if (spec.kind != sf::LayerKind::kConv) continue;
    if (!requested_layers.empty() && !requested_layers.count(spec.L)) continue;

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

    layer.OverrideWeightCache(cache_cfg);
    layer.run_layer();

    const auto* cache_stats = layer.weight_cache_stats();
    if (!cache_stats) {
      std::cerr << "Weight cache stats unavailable." << std::endl;
      return 2;
    }

    const std::uint64_t total_demand =
        cache_stats->demand_hits +
        cache_stats->demand_misses_allocated +
        cache_stats->demand_misses_noalloc;
    const double hit_rate =
        (total_demand == 0)
            ? 0.0
            : static_cast<double>(cache_stats->demand_hits) /
                  static_cast<double>(total_demand);

    std::cout << spec.L << ','
              << point.capacity_kb << ','
              << point.ways << ','
              << num_sets << ','
              << cache_stats->latency_cycles << ','
              << cache_stats->demand_hits << ','
              << cache_stats->demand_misses_allocated << ','
              << cache_stats->demand_misses_noalloc << ','
              << cache_stats->prefetch_hits << ','
              << hit_rate << '\n';
    csv_ofs << spec.L << ','
            << point.capacity_kb << ','
            << point.ways << ','
            << num_sets << ','
            << cache_stats->latency_cycles << ','
            << cache_stats->demand_hits << ','
            << cache_stats->demand_misses_allocated << ','
            << cache_stats->demand_misses_noalloc << ','
            << cache_stats->prefetch_hits << ','
            << hit_rate << '\n';
  }
  }

  return 0;
}
