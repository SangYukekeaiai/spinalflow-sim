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
  bool        trace_enabled   = true;     // enable/disable any trace output
  std::string trace_output_path;          // optional path for detailed trace output
  std::size_t trace_max_lines = 0;        // max lines to emit (0 = unlimited)
  // Toggle original modulo-based set/tag mapping vs. hashed indexing.
  // false (default): hashed index + full-key tag
  // true:           original index (key % sets) + tag (key / sets)
  bool use_original_maptotag = false;
  // Independent toggle for ts_duration CSV emission. When true, per-layer
  // timestep-access CSVs are written even if tracing is disabled.
  bool        ts_duration_enabled = false;
  // Optional absolute/relative path to the stats/<repo>/<model> directory to
  // derive ts_duration output paths when trace_output_path is empty.
  std::string stats_model_dir;
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
  // Per-set unique demand lines encountered (key = set index)
  std::unordered_map<int, std::uint64_t> per_set_unique_demand_lines;
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
  for (const auto& [set_idx, count_a] : a.per_set_unique_demand_lines) {
    const auto it_b = b.per_set_unique_demand_lines.find(set_idx);
    const std::uint64_t count_b =
        (it_b != b.per_set_unique_demand_lines.end()) ? it_b->second : 0ULL;
    if (count_a > count_b) {
      d.per_set_unique_demand_lines.emplace(set_idx, count_a - count_b);
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
  // Add counts from an external snapshot (channel_id -> delta_count)
  void MergeAdd(const std::unordered_map<int, int>& other) {
    for (const auto& kv : other) {
      scores_[kv.first] += kv.second;
    }
  }
  // Replace scores with an external snapshot (channel_id -> count)
  void Assign(const std::unordered_map<int, int>& other) { scores_ = other; }
  // Snapshot of current per-channel scores (channel_id -> score)
  std::unordered_map<int, int> Snapshot() const { return scores_; }
private:
  std::unordered_map<int, int> scores_;
};

//------------------------------------------------------------------------------
// Simple set-associative cache simulator for latency accounting
//------------------------------------------------------------------------------
class CacheSim {
public:
  explicit CacheSim(const CacheConfig& cfg);
  void Reset();
  // Notify that a spike occurs on input channel 'cin' to bias future evictions.
  void NotifySpike(int cin);
  // Begin a new time step t. At t=0, eviction uses LRU-only while
  // S[0] is accumulated. For t>=1, eviction uses S[t-1] while S[t]
  // is accumulated via NotifySpike(). Repeated calls with the same t
  // are ignored.
  void BeginTimeStep(int t);

  // Access a demand line. Returns metrics for this access (miss/hit, latency, prefetch work).
  // This may trigger sequential prefetches for cin+1..cin+prefetch_depth.
  AccessResult Access(const LineAddr& la);
  AccessResult AccessLRU(const LineAddr& la);
  AccessResult AccessWithPolicy(const LineAddr& la, EvictionPolicy policy);

  CacheStats GetStats() const { return stats_; }

  // Optional helpers
  int NumSets() const { return num_sets_; }
  const CacheConfig& Config() const { return cfg_; }
  // Expose a snapshot of the cumulative scoreboard over the whole layer run
  // by summing S[t] across all observed timesteps.
  std::unordered_map<int, int> ScoreboardSnapshot() const {
    std::unordered_map<int, int> accum;
    for (const auto& kv : scoreboard_by_t_) {
      const auto snap = kv.second.Snapshot();
      for (const auto& p : snap) {
        accum[p.first] += p.second;
      }
    }
    return accum;
  }

  // Context hooks for per-site timestep accounting and summary.
  // Set the current output site id (spine id = h_out * W_out + w_out).
  void SetCurrentOutputSpine(int spine_id) { current_spine_id_ = spine_id; }
  // Provide layer dims so we can emit a one-line summary with config.
  void SetLayerDims(int Cin, int Hin, int Win,
                    int Cout, int Hout, int Wout,
                    int Kh, int Kw) {
    layer_Cin_  = Cin;  layer_Hin_  = Hin;  layer_Win_  = Win;
    layer_Cout_ = Cout; layer_Hout_ = Hout; layer_Wout_ = Wout;
    layer_Kh_   = Kh;   layer_Kw_   = Kw;
  }
  // Provide current layer id for path building of scoreboard/timestep CSVs.
  void SetLayerId(int L) { layer_id_for_paths_ = L; }

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
  // Original modulo-based mapping (index=key%sets, tag=key/sets)
  std::pair<int, uint64_t> MapToSetTag(uint64_t key) const;
  // Hashed-index mapping (index=xorfold(key, sets, channel), tag=key)
  std::pair<int, uint64_t> MapToSetTag(uint64_t key, int channel_id) const;
  // Unified selector: choose mapping based on a boolean toggle.
  std::pair<int, uint64_t> MapToSetTag(uint64_t key, int channel_id, bool use_original) const;
  int  FindHit(Set& set, uint64_t tag) const;
  void TouchLRU(Set& set, int way);
  int  PickVictim(int set_idx, Set& set, EvictionPolicy policy);
  int  PickVictimScoreboard(int set_idx, Set& set);
  int  PickVictimLRU(int set_idx, Set& set);
  bool InSameTile(const LineAddr& a, const LineAddr& b) const;
  void WriteTrace(const std::string& message);
  bool TraceHasCapacity() const;

private:
  CacheConfig      cfg_;
  int              num_sets_ = 0;
  std::vector<Set> sets_;
  // Per-timestep scoreboard state, accumulated across the entire layer:
  // S_total[t] is updated online as spikes occur at timestep t.
  // Aggregated per-timestep scoreboards across the entire layer run.
  // Key: timestep t -> cumulative S_total[t]
  std::unordered_map<int, Scoreboard> scoreboard_by_t_;
  int              current_timestep_ = -1; // unknown
  bool             use_lru_this_step_ = true; // LRU-only at t=0
  CacheStats       stats_{};
  std::unordered_set<uint64_t> unique_demand_lines_seen_;
  std::unordered_map<uint64_t, std::uint64_t> last_access_turn_;
  std::unordered_map<uint64_t, int> last_access_timestep_;
  std::uint64_t access_sequence_counter_ = 0;
  std::unique_ptr<std::ofstream> trace_stream_;
  std::size_t trace_lines_written_ = 0;
  
  // --- Per-timestep access counting (per output site) ---
  // Counts demand accesses per (output_spine_id, timestep).
  // Keyed by site_id -> (timestep -> count)
  std::unordered_map<int, std::unordered_map<int, std::uint64_t>> per_site_step_access_counts_;
  int max_timestep_observed_ = -1;
  int current_spine_id_ = -1;

  // Layer config snapshot for summary line
  int layer_Cin_ = 0, layer_Hin_ = 0, layer_Win_ = 0;
  int layer_Cout_ = 0, layer_Hout_ = 0, layer_Wout_ = 0;
  int layer_Kh_ = 0, layer_Kw_ = 0;
  int layer_id_for_paths_ = -1;
};

void PrintCacheConfig(const CacheConfig& cfg);

} // namespace sf::arch::cache
