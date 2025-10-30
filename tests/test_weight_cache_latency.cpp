// All comments are in English.
#include "cache/cache_iface.h"
#include "cache/cache_config.h"
#include "cache/registry.h"

#include <cassert>
#include <iostream>

int main() {
  sf::cache::CacheConfig cfg;
  cfg.geometry.num_sets = 16;
  cfg.geometry.ways = 1;
  cfg.geometry.line_size_bytes = 128;
  cfg.timing.hit_latency_cycles = 1;
  cfg.timing.miss_latency_cycles = 64;
  cfg.A1 = 1;
  cfg.Cin = 2;
  cfg.KH = 2;
  cfg.KW = 2;
  cfg.Validate();

  auto modules = sf::cache::MakeDefaultModules(cfg);
  auto cache = sf::cache::BuildCache(cfg, std::move(modules));

  cache->SetWindow(0, 1);

  for (int cin = 0; cin < cfg.Cin; ++cin) {
    for (int kh = 0; kh < cfg.KH; ++kh) {
      for (int kw = 0; kw < cfg.KW; ++kw) {
        sf::cache::AccessRequest request;
        request.tile_id = 0;
        request.cin = cin;
        request.kh = kh;
        request.kw = kw;
        auto res = cache->OnDemandAccess(request);
        assert(!res.hit && "First tile should miss and allocate.");
        assert(res.allocated);
        assert(res.window_admitted);
      }
    }
  }

  const auto& stats_after_tile0 = cache->Stats();
  assert(stats_after_tile0.demand_misses_allocated == 8);
  assert(stats_after_tile0.demand_hits == 0);
  assert(stats_after_tile0.prefetch_inserts == 8);

  cache->SetWindow(1, 2);

  for (int cin = 0; cin < cfg.Cin; ++cin) {
    for (int kh = 0; kh < cfg.KH; ++kh) {
      for (int kw = 0; kw < cfg.KW; ++kw) {
        sf::cache::AccessRequest request;
        request.tile_id = 1;
        request.cin = cin;
        request.kh = kh;
        request.kw = kw;
        auto res = cache->OnDemandAccess(request);
        assert(res.hit && "Prefetched line should hit for next tile.");
        assert(!res.allocated);
      }
    }
  }

  const auto& stats = cache->Stats();
  assert(stats.demand_hits == 8);
  assert(stats.latency_cycles == 8 * cfg.timing.miss_latency_cycles + 8 * cfg.timing.hit_latency_cycles);
  std::cout << "Weight cache total latency cycles: " << stats.latency_cycles << "\n";
  std::cout << "Demand bytes loaded: " << stats.demand_bytes_loaded << "\n";
  std::cout << "Prefetch bytes loaded: " << stats.prefetch_bytes_loaded << "\n";
  std::cout << "Evictions observed: " << stats.evictions_total << "\n";
  return 0;
}
