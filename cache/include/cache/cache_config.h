// All comments are in English.
#pragma once

#include <cstdint>
#include <stdexcept>

namespace sf::cache {

struct CacheGeometry {
  int num_sets = 0;          // total cache sets
  int ways = 0;              // associativity per set
  int line_size_bytes = 128; // bytes per line
};

struct CacheTiming {
  std::uint64_t hit_latency_cycles = 1;
  std::uint64_t miss_latency_cycles = 40;
};

enum class ReplacementKind {
  Lru = 0,
  Random
};

struct CacheConfig {
  CacheGeometry geometry{};
  CacheTiming timing{};
  int Cin = 0;
  int KH = 0;
  int KW = 0;
  ReplacementKind replacement_kind = ReplacementKind::Lru;

  void Validate() const {
    if (Cin <= 0 || KH <= 0 || KW <= 0) {
      throw std::invalid_argument("CacheConfig: Cin, KH, and KW must be positive.");
    }
    if (geometry.num_sets <= 0) {
      throw std::invalid_argument("CacheConfig: num_sets must be positive.");
    }
    if (geometry.ways <= 0) {
      throw std::invalid_argument("CacheConfig: ways must be positive.");
    }
    if (geometry.line_size_bytes <= 0) {
      throw std::invalid_argument("CacheConfig: line_size_bytes must be positive.");
    }
  }
};

CacheConfig MakeDefaultConfig(int Cin, int KH, int KW);

} // namespace sf::cache
