#pragma once
// All comments are in English.

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <ostream>
#include <string>
#include <memory>
#include <fstream>

namespace sf::arch::cache {

enum class EvictionPolicy {
  kScoreboard,
  kLRU
};

//------------------------------------------------------------------------------
// Configuration for the simple weight-cache latency model
//------------------------------------------------------------------------------
struct CacheConfig {
  std::size_t capacity_bytes = 576 * 1024; // total cache size
  std::size_t line_bytes     = 128;       // one cache line = one DRAM line
  int         ways           = 8;         // set associativity
  int         l1_hit_cycles  = 1;         // cycles to serve a hit
  int         miss_overhead  = 40;        // fixed per-line miss penalty (cycles)
  int         prefetch_depth = 0;         // demand for (cin) triggers prefetch for cin+1..+N
  EvictionPolicy eviction_policy = EvictionPolicy::kScoreboard;
  std::string trace_output_path;          // optional path for detailed trace output
  std::size_t trace_max_lines = 0;        // max lines to emit (0 = unlimited)
};

//------------------------------------------------------------------------------
// Address for one 128B weight line.
// Note: layer dimension is intentionally omitted per user's request.
//------------------------------------------------------------------------------
struct LineAddr {
  uint32_t tile = 0; // tile id
  uint32_t cin  = 0; // input channel id
  uint32_t kh   = 0; // kernel height index
  uint32_t kw   = 0; // kernel width index
  uint64_t key  = 0; // packed key derived from the 4 indices

  LineAddr() = default;
  LineAddr(uint32_t t, uint32_t c, uint32_t h, uint32_t w)
      : tile(t), cin(c), kh(h), kw(w), key(ComposeKey(t, c, h, w)) {}

  // Pack (tile, cin, kh, kw) into a stable 64-bit key (no hashing).
  // Bit layout: [tile:24][cin:16][kh:12][kw:12] = 64 bits
  static inline uint64_t ComposeKey(uint32_t tile, uint32_t cin,
                                    uint32_t kh,   uint32_t kw) {
    const uint64_t T = (static_cast<uint64_t>(tile) & 0xFFFFFFull) << 40;
    const uint64_t C = (static_cast<uint64_t>(cin)  & 0xFFFFull)   << 24;
    const uint64_t H = (static_cast<uint64_t>(kh)   & 0xFFFull)    << 12;
    const uint64_t W = (static_cast<uint64_t>(kw)   & 0xFFFull);
    return (T | C | H | W);
  }
};

//------------------------------------------------------------------------------
// Result of a single demand access (plus any prefetch work triggered by it)
//------------------------------------------------------------------------------
struct AccessResult {
  int  demand_cycles       = 0;
  bool demand_miss         = false;
  int  prefetch_requests   = 0;
  int  prefetch_miss_lines = 0;
};

//------------------------------------------------------------------------------
// Accumulated statistics for cache accesses
//------------------------------------------------------------------------------
struct CacheStats {
  std::uint64_t demand_accesses    = 0;
  std::uint64_t demand_misses      = 0;
  std::uint64_t demand_hit_cycles  = 0;
  std::uint64_t demand_miss_cycles = 0;
  std::uint64_t prefetch_requests  = 0;
  std::uint64_t prefetch_misses    = 0;
  std::uint64_t unique_demand_lines = 0;
  std::uint64_t zero_score_events  = 0;
  std::uint64_t reuse_distance_total = 0;
  std::uint64_t reuse_events         = 0;
  std::unordered_map<std::uint64_t, std::uint64_t> reuse_distance_histogram;
};

inline CacheStats operator-(const CacheStats& a, const CacheStats& b) {
  CacheStats d{};
  d.demand_accesses    = (a.demand_accesses    >= b.demand_accesses)    ? (a.demand_accesses    - b.demand_accesses)    : 0;
  d.demand_misses      = (a.demand_misses      >= b.demand_misses)      ? (a.demand_misses      - b.demand_misses)      : 0;
  d.demand_hit_cycles  = (a.demand_hit_cycles  >= b.demand_hit_cycles)  ? (a.demand_hit_cycles  - b.demand_hit_cycles)  : 0;
  d.demand_miss_cycles = (a.demand_miss_cycles >= b.demand_miss_cycles) ? (a.demand_miss_cycles - b.demand_miss_cycles) : 0;
  d.prefetch_requests  = (a.prefetch_requests  >= b.prefetch_requests)  ? (a.prefetch_requests  - b.prefetch_requests)  : 0;
  d.prefetch_misses    = (a.prefetch_misses    >= b.prefetch_misses)    ? (a.prefetch_misses    - b.prefetch_misses)    : 0;
  d.unique_demand_lines = (a.unique_demand_lines >= b.unique_demand_lines)
                              ? (a.unique_demand_lines - b.unique_demand_lines)
                              : 0;
  d.zero_score_events  = (a.zero_score_events  >= b.zero_score_events)
                             ? (a.zero_score_events - b.zero_score_events)
                             : 0;
  d.reuse_distance_total = (a.reuse_distance_total >= b.reuse_distance_total)
                              ? (a.reuse_distance_total - b.reuse_distance_total)
                              : 0;
  d.reuse_events = (a.reuse_events >= b.reuse_events)
                       ? (a.reuse_events - b.reuse_events)
                       : 0;
  for (const auto& [distance, count_a] : a.reuse_distance_histogram) {
    const auto it_b = b.reuse_distance_histogram.find(distance);
    const std::uint64_t count_b =
        (it_b != b.reuse_distance_histogram.end()) ? it_b->second : 0ULL;
    if (count_a > count_b) {
      d.reuse_distance_histogram.emplace(distance, count_a - count_b);
    }
  }
  return d;
}

//------------------------------------------------------------------------------
// Simple per-channel scoreboard to bias eviction toward cooler channels
//------------------------------------------------------------------------------
class Scoreboard {
public:
  void Bump(int channel_id)            { scores_[channel_id]++; }
  int  Get(int channel_id) const;
  void Dump(std::ostream& os) const;
  void Clear()                         { scores_.clear(); }
private:
  std::unordered_map<int, int> scores_;
};

//------------------------------------------------------------------------------
// Simple set-associative cache simulator for latency accounting
//------------------------------------------------------------------------------
class CacheSim {
public:
  explicit CacheSim(const CacheConfig& cfg);
  int              tmpZeroScoreCount_ = 0;
  void Reset();
  // Notify that a spike occurs on input channel 'cin' to bias future evictions.
  void NotifySpike(int cin);

  // Access a demand line. Returns metrics for this access (miss/hit, latency, prefetch work).
  // This may trigger sequential prefetches for cin+1..cin+prefetch_depth.
  AccessResult Access(const LineAddr& la);
  AccessResult AccessLRU(const LineAddr& la);
  AccessResult AccessWithPolicy(const LineAddr& la, EvictionPolicy policy);

  CacheStats GetStats() const { return stats_; }

  // Optional helpers
  int NumSets() const { return num_sets_; }
  const CacheConfig& Config() const { return cfg_; }

private:
  struct WayEntry {
    uint64_t tag       = 0;
    bool     valid     = false;
    int      lru_counter = 0;  // increasing "age"; 0 on touch; larger => older
    int      channel_id = -1;  // cached line's input channel (cin)
  };

  struct Set {
    std::vector<WayEntry> ways;
  };

  struct ServeResult {
    int  cycles = 0;
    bool miss   = false;
  };

  ServeResult ServeOne(const LineAddr& la, bool is_prefetch, EvictionPolicy policy);
  std::pair<int, uint64_t> MapToSetTag(uint64_t key) const;
  int  FindHit(Set& set, uint64_t tag) const;
  void TouchLRU(Set& set, int way);
  int  PickVictim(Set& set, EvictionPolicy policy);
  int  PickVictimScoreboard(Set& set);
  int  PickVictimLRU(Set& set);
  bool InSameTile(const LineAddr& a, const LineAddr& b) const;
  void WriteTrace(const std::string& message);
  bool TraceAvailable() const;
  bool TraceHasCapacity() const;

private:
  CacheConfig      cfg_;
  int              num_sets_ = 0;
  std::vector<Set> sets_;
  Scoreboard       scoreboard_;
  CacheStats       stats_{};
  std::unordered_set<uint64_t> unique_demand_lines_seen_;
  std::unordered_map<uint64_t, std::uint64_t> last_access_turn_;
  std::uint64_t access_sequence_counter_ = 0;
  std::unique_ptr<std::ofstream> trace_stream_;
  std::size_t trace_lines_written_ = 0;
  
};

void PrintCacheConfig(const CacheConfig& cfg);

} // namespace sf::arch::cache
