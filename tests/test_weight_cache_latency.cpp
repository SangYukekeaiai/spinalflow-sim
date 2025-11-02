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
  cfg.Cin = 2;
  cfg.KH = 2;
  cfg.KW = 2;
  cfg.Validate();

  auto modules = sf::cache::MakeDefaultModules(cfg);
  auto cache = sf::cache::BuildCache(cfg, std::move(modules));

  auto access_tile = [&](int tile_id) {
    for (int cin = 0; cin < cfg.Cin; ++cin) {
      for (int kh = 0; kh < cfg.KH; ++kh) {
        for (int kw = 0; kw < cfg.KW; ++kw) {
          sf::cache::AccessRequest request;
          request.tile_id = tile_id;
          request.cin = cin;
          request.kh = kh;
          request.kw = kw;
          (void)cache->OnDemandAccess(request);
        }
      }
    }
  };

  cache->SetWindow(0, 1);
  access_tile(0);

  const auto& stats_after_tile0 = cache->Stats();
  const int total_lines = cfg.Cin * cfg.KH * cfg.KW;
  assert(stats_after_tile0.demand_misses_allocated == static_cast<std::uint64_t>(total_lines));
  assert(stats_after_tile0.demand_hits == 0);
  assert(stats_after_tile0.prefetch_inserts == static_cast<std::uint64_t>(total_lines));
  assert(stats_after_tile0.prefetch_hits == 0);

  cache->SetWindow(1, 0);
  access_tile(1);

  const auto& stats = cache->Stats();
  assert(stats.demand_hits == static_cast<std::uint64_t>(total_lines));
  assert(stats.demand_misses_allocated == static_cast<std::uint64_t>(total_lines));
  assert(stats.prefetch_inserts == static_cast<std::uint64_t>(total_lines));
  assert(stats.prefetch_hits == static_cast<std::uint64_t>(total_lines));
  const auto expected_latency =
      static_cast<std::uint64_t>(total_lines) * (cfg.timing.miss_latency_cycles + cfg.timing.hit_latency_cycles);
  assert(stats.latency_cycles == expected_latency);
  std::cout << "Weight cache total latency cycles: " << stats.latency_cycles << "\n";
  std::cout << "Demand bytes loaded: " << stats.demand_bytes_loaded << "\n";
  std::cout << "Prefetch bytes loaded: " << stats.prefetch_bytes_loaded << "\n";
  std::cout << "Evictions observed: " << stats.evictions_total << "\n";
  return 0;
}
