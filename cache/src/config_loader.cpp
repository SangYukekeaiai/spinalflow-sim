// All comments are in English.
#include "cache/cache_config.h"

namespace sf::cache {

CacheConfig MakeDefaultConfig(int Cin, int KH, int KW) {
  CacheConfig cfg;
  cfg.Cin = Cin;
  cfg.KH = KH;
  cfg.KW = KW;
  cfg.geometry.num_sets = 128;
  cfg.geometry.ways = 4;
  cfg.geometry.line_size_bytes = 128;
  cfg.timing.hit_latency_cycles = 1;
  cfg.timing.miss_latency_cycles = 128;
  cfg.A1 = 1;
  cfg.Validate();
  return cfg;
}

} // namespace sf::cache

