// All comments are in English.
#pragma once

#include <cstdint>

#include "cache/cache_config.h"
#include "cache/cache_iface.h"

namespace test::stats {

// Captures derived, per-layer metrics without polluting core cache logic.
struct LayerStatsSummary {
  int layer = -1;
  std::uint64_t demand_accesses = 0;
  std::uint64_t hits = 0;
  std::uint64_t misses = 0;
  std::uint64_t hit_cycles = 0;
  std::uint64_t miss_cycles = 0;
  std::uint64_t total_cycles = 0;
  double hit_rate = 0.0;
  std::uint64_t unique_demand_lines = 0;

  static LayerStatsSummary FromCacheStats(int layer,
                                          const sf::cache::CacheStats& stats,
                                          const sf::cache::CacheConfig& cfg) {
    LayerStatsSummary summary;
    summary.layer = layer;
    summary.hits = stats.demand_hits;
    summary.misses = stats.demand_misses_allocated + stats.demand_misses_noalloc;
    summary.demand_accesses = summary.hits + summary.misses;
    summary.hit_cycles = stats.demand_hits * cfg.timing.hit_latency_cycles;
    const std::uint64_t miss_total = stats.demand_misses_allocated + stats.demand_misses_noalloc;
    summary.miss_cycles = miss_total * cfg.timing.miss_latency_cycles;
    summary.total_cycles = stats.latency_cycles;
    summary.hit_rate = (summary.demand_accesses == 0)
                           ? 0.0
                           : static_cast<double>(summary.hits) /
                                 static_cast<double>(summary.demand_accesses);
    summary.unique_demand_lines = stats.demand_misses_allocated;
    return summary;
  }
};

} // namespace test::stats
