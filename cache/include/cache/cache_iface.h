// All comments are in English.
#pragma once

#include <cstdint>
#include <memory>
#include <optional>

#include "cache/cache_config.h"

namespace sf::cache {

struct AccessRequest {
  int tile_id = -1;
  int cin = -1;
  int kh = -1;
  int kw = -1;
  int output_spine_id = -1;
  int timestep = -1;
};

struct AccessResult {
  bool hit = false;
  bool allocated = false;
  bool window_admitted = false;
  std::uint64_t bytes_fetched = 0;
  std::uint64_t latency_cycles = 0;
  int set_idx = -1;
  int way = -1;
  int tile_id = -1;
  int L = -1;
  std::uint64_t tag = 0;
  bool evicted = false;
  std::uint64_t evicted_tag = 0;
  int evicted_tile_id = -1;
  int evicted_output_spine_id = -1;
  int evicted_last_timestep = -1;
};

struct CacheStats {
  std::uint64_t demand_hits = 0;
  std::uint64_t demand_misses_allocated = 0;
  std::uint64_t demand_misses_noalloc = 0;
  std::uint64_t prefetch_hits = 0;
  std::uint64_t prefetch_inserts = 0;
  std::uint64_t evictions_total = 0;
  std::uint64_t latency_cycles = 0;
  std::uint64_t demand_bytes_loaded = 0;
  std::uint64_t prefetch_bytes_loaded = 0;
};

class ICache {
public:
  virtual ~ICache() = default;
  virtual void Reset() = 0;
  virtual void SetWindow(int cur_tile, int next_tile) = 0;
  virtual void AdvanceWindow() = 0;
  virtual AccessResult OnDemandAccess(const AccessRequest& request) = 0;
  virtual const CacheStats& Stats() const = 0;
};

struct CacheModules;

std::unique_ptr<ICache> BuildCache(const CacheConfig& cfg,
                                   CacheModules modules);

} // namespace sf::cache
