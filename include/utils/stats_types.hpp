// All comments are in English.
#pragma once

#include <string>
#include <vector>
#include <unordered_map>

#include "core/core.hpp"            // CoreCycleStats, CoreSramStats
#include "arch/cache/cache.hpp"     // CacheStats
#include "runner/simulation.hpp"    // LayerKind

namespace sf {

// Lightweight record to aggregate per-layer stats for CSV emission.
struct LayerStageRecord {
  int layer_id = 0;
  std::string layer_name;
  LayerKind kind = LayerKind::kConv;
  CoreCycleStats cycles{};
  CoreSramStats sram_stats{};
  sf::arch::cache::CacheStats cache_stats{};
  // Snapshot of scoreboard scores for this layer (channel_id -> score)
  std::unordered_map<int, int> scoreboard_scores{};
};

// Aggregated cache totals row used for CSV emission
struct CacheTotalsRow {
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

} // namespace sf
