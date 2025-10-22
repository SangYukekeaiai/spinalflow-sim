// All comments are in English.

#include "arch/cache/cache.hpp"
#include "utils/stats_io.hpp"

#include <climits>
#include <iostream>
#include <filesystem>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <unordered_set>

namespace sf::arch::cache {

namespace {
// XOR-fold hash indexing: fold higher tag bits into the index to
// break power-of-two/regular strides. Works for any number of sets.
inline uint32_t mix64(uint64_t x) {
  x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33; return static_cast<uint32_t>(x);
}

inline uint32_t index_xorfold(uint64_t line_addr, uint32_t num_sets, uint32_t channel_id) {
  // Fold some higher bits and a per-channel salt
  const uint64_t folded = line_addr ^ (line_addr >> 11) ^ (static_cast<uint64_t>(channel_id) * 0x9e3779b97f4a7c15ULL);
  const uint32_t h = mix64(folded);
  if (num_sets == 0u) return 0u;
  // If num_sets is a power of two, use a mask; otherwise use modulo
  return ((num_sets & (num_sets - 1u)) == 0u) ? (h & (num_sets - 1u)) : (h % num_sets);
}
} // anonymous namespace

//-------------------------- Scoreboard ----------------------------------------

int Scoreboard::Get(int channel_id) const {
  auto it = scores_.find(channel_id);
  return (it == scores_.end()) ? 0 : it->second;
}

void Scoreboard::Dump(std::ostream& os) const {
  os << "[Scoreboard]";
  if (scores_.empty()) {
    os << " empty\n";
    return;
  }
  // Collect and sort by score (ascending), then by channel id (ascending)
  std::vector<std::pair<int,int>> items;
  items.reserve(scores_.size());
  for (const auto& kv : scores_) items.emplace_back(kv.first, kv.second);
  std::sort(items.begin(), items.end(), [](const auto& a, const auto& b){
    if (a.second != b.second) return a.second < b.second; // by score
    return a.first < b.first; // tie-break by channel id
  });
  bool first = true;
  for (const auto& [channel, score] : items) {
    os << (first ? " " : ", ");
    os << "(cin=" << channel << ", score=" << score << ")";
    first = false;
  }
  os << '\n';
}

//--------------------------- CacheSim -----------------------------------------

CacheSim::CacheSim(const CacheConfig& cfg) : cfg_(cfg) {
  // Compute number of sets = (total lines) / ways
  const std::size_t total_lines = cfg_.capacity_bytes / cfg_.line_bytes;
  num_sets_ = static_cast<int>(total_lines / static_cast<std::size_t>(cfg_.ways));
  // std::cout << "CacheSim: capacity=" << cfg_.capacity_bytes
  //           << " bytes, line=" << cfg_.line_bytes
  //           << " bytes, ways=" << cfg_.ways
  //           << ", sets=" << num_sets_ << "\n";
  if (num_sets_ <= 0) num_sets_ = 1; // guard against degenerate configs
  sets_.resize(static_cast<std::size_t>(num_sets_), Set{std::vector<WayEntry>(cfg_.ways)});

  if (!cfg_.trace_output_path.empty()) {
    const std::filesystem::path trace_path(cfg_.trace_output_path);
    if (!trace_path.parent_path().empty()) {
      std::filesystem::create_directories(trace_path.parent_path());
    }
    trace_stream_ = std::make_unique<std::ofstream>(trace_path, std::ios::out | std::ios::trunc);
    if (!trace_stream_ || !trace_stream_->is_open()) {
      throw std::runtime_error("CacheSim: failed to open trace file " + trace_path.string());
    }
  }
}

void CacheSim::Reset() {
  for (Set& set : sets_) {
    for (WayEntry& way : set.ways) {
      way.valid = false;
      way.lru_counter = 0;
      way.tag = 0;
      way.channel_id = -1;
    }
  }
  // Emit per-(tile, output_spine, timestep) Hit/Cold/Conflict CSV next to trace
  try {
    sf::WriteTileHitColdConflictCsv(
        cfg_,
        layer_id_for_paths_,
        layer_Wout_,
        per_tile_site_step_counts_);
  } catch (...) {
    // swallow errors to keep simulation robust
  }
  // Emit cumulative per-step scoreboard CSVs (S_total[t] for each observed t)
  for (const auto& kv : scoreboard_by_t_) {
    const int t = kv.first;
    const auto snapshot = kv.second.Snapshot();
    sf::WriteScoreboardStepCsvIfEnabled(cfg_, layer_id_for_paths_, t, snapshot);
  }
  // Emit per-site timestep access counts for the completed layer/config
  sf::WriteLayerTimestepAccessCsvsIfEnabled(
      cfg_,
      per_site_step_access_counts_,
      max_timestep_observed_,
      layer_id_for_paths_,
      layer_Cin_, layer_Hin_, layer_Win_);
  scoreboard_by_t_.clear();
  current_timestep_ = -1;
  use_lru_this_step_ = true;
  unique_demand_lines_seen_.clear();
  last_access_turn_.clear();
  last_access_timestep_.clear();
  access_sequence_counter_ = 0;
  stats_ = {};
  per_site_step_access_counts_.clear();
  per_tile_site_step_counts_.clear();
  max_timestep_observed_ = -1;
  current_spine_id_ = -1;
  if (trace_stream_) {
    trace_stream_->flush();
  }
}

void CacheSim::NotifySpike(int cin) {
  const int t = (current_timestep_ < 0) ? 0 : current_timestep_;
  scoreboard_by_t_[t].Bump(cin);
}

void CacheSim::BeginTimeStep(int t) {
  // Ignore redundant calls for the same time step
  if (current_timestep_ == t) {
    return;
  }
  if (current_timestep_ < 0) {
    // First observed time step
    current_timestep_ = t;
    // LRU-only eviction for t==0 while we build S_total[0]
    use_lru_this_step_ = (t == 0);
    return;
  }
  // Transition to a new timestep: decide eviction mode only
  if (t == 0) {
    // LRU-only eviction when time wraps/restarts at 0
    use_lru_this_step_ = true;
  } else {
    use_lru_this_step_ = false;
  }
  current_timestep_ = t;
}

AccessResult CacheSim::Access(const LineAddr& la) {
  return AccessWithPolicy(la, cfg_.eviction_policy);
}

AccessResult CacheSim::AccessLRU(const LineAddr& la) {
  return AccessWithPolicy(la, EvictionPolicy::kLRU);
}

AccessResult CacheSim::AccessWithPolicy(const LineAddr& la, EvictionPolicy policy) {
  AccessResult out{};

  // Serve the demand line
  const ServeResult demand = ServeOne(la, /*is_prefetch=*/false, policy);
  out.demand_cycles = demand.cycles;
  out.demand_miss   = demand.miss;

  if (unique_demand_lines_seen_.insert(la.key).second) {
    stats_.unique_demand_lines++;
    // Also attribute this unique line to its set index
    const auto mapped = MapToSetTag(la.key, static_cast<int>(la.cin), cfg_.use_original_maptotag);
    const int uc_set_idx = mapped.first;
    stats_.per_set_unique_demand_lines[uc_set_idx] += 1ULL;
  }

  stats_.demand_accesses++;
  access_sequence_counter_++;
  const std::uint64_t current_turn = access_sequence_counter_;
  auto [last_it, inserted] = last_access_turn_.emplace(la.key, current_turn);
  if (!inserted) {
    const std::uint64_t distance = current_turn - last_it->second;
    stats_.reuse_distance_total += distance;
    stats_.reuse_events++;
    stats_.reuse_distance_histogram[distance]++;
  }
  last_it->second = current_turn;
  // Track the last timestep this line was accessed (for eviction duration reporting)
  if (current_timestep_ < 0) {
    current_timestep_ = 0;
  }
  last_access_timestep_[la.key] = current_timestep_;
  if (demand.miss) {
    stats_.demand_misses++;
    stats_.demand_miss_cycles += static_cast<std::uint64_t>(demand.cycles);
  } else {
    stats_.demand_hit_cycles += static_cast<std::uint64_t>(demand.cycles);
  }

  // Count demand accesses for the current timestep per output site (spine id)
  if (current_timestep_ < 0) {
    // If BeginTimeStep(t) wasn't called yet, treat as t=0
    current_timestep_ = 0;
  }
  if (current_spine_id_ >= 0) {
    per_site_step_access_counts_[current_spine_id_][current_timestep_] += 1ULL;
    if (current_timestep_ > max_timestep_observed_) {
      max_timestep_observed_ = current_timestep_;
    }
  }

  // Sequentially issue simple next-channel prefetches up to prefetch_depth
  if (demand.miss) {
    for (int d = 1; d <= cfg_.prefetch_depth; ++d) {
      LineAddr pf(la.tile, la.cin + static_cast<uint32_t>(d), la.kh, la.kw);
      if (!InSameTile(la, pf)) break; // only prefetch within the same tile/kh/kw group
      const ServeResult pf_res = ServeOne(pf, /*is_prefetch=*/true, policy);
      out.prefetch_requests++;
      stats_.prefetch_requests++;
      if (pf_res.miss) {
        out.prefetch_miss_lines++;
        stats_.prefetch_misses++;
      }
    }
  }

  return out;
}

CacheSim::ServeResult CacheSim::ServeOne(const LineAddr& la, bool is_prefetch, EvictionPolicy policy) {
  ServeResult result{};
  // Map to set + tag
  auto [set_idx, tag] = MapToSetTag(la.key, static_cast<int>(la.cin), cfg_.use_original_maptotag);
  const char* access_kind = is_prefetch ? "PF" : "DM";
  Set& set = sets_[static_cast<std::size_t>(set_idx)];
  // Derive output location if available
  int hout = -1, wout = -1;
  if (layer_Wout_ > 0 && current_spine_id_ >= 0) {
    hout = current_spine_id_ / layer_Wout_;
    wout = current_spine_id_ % layer_Wout_;
  }

  // Check hit
  const int hit_way = FindHit(set, tag);
  if (hit_way >= 0) {
    if (TraceHasCapacity()) {
      std::ostringstream oss;
      oss << "[CacheSim][" << access_kind << "][HIT]";
      if (!is_prefetch) {
        oss << " acc=" << (stats_.demand_accesses + 1);
      }
      oss << " tile=" << la.tile;
      if (hout >= 0) {
        oss << " hout=" << hout << " wout=" << wout;
      }
      oss << " t=" << current_timestep_
          << " set=" << set_idx
          << " way=" << hit_way
          << " key=" << la.key
          << " cin=" << la.cin
          << " kh=" << la.kh
          << " kw=" << la.kw;
      WriteTrace(oss.str());
    }
    // Count DM hits by (tile, output site, timestep)
    if (!is_prefetch && current_spine_id_ >= 0) {
      const int t = (current_timestep_ < 0) ? 0 : current_timestep_;
      per_tile_site_step_counts_[static_cast<int>(la.tile)][current_spine_id_][t].hits += 1ULL;
    }
    TouchLRU(set, hit_way);
    result.cycles = is_prefetch ? 0 : cfg_.l1_hit_cycles;
    result.miss = false;
    return result;
  }

  // Decide the victim first so we can classify the miss
  const int vic = PickVictim(set_idx, set, policy);
  WayEntry& victim_entry = set.ways[static_cast<std::size_t>(vic)];
  const bool will_evict = (victim_entry.valid && victim_entry.channel_id >= 0);

  if (TraceHasCapacity()) {
    std::ostringstream oss;
    oss << "[CacheSim][" << access_kind << "][MISS]";
    // Tag miss type: [COLD] if it will fill an empty way, [CONFLICT] if it triggers eviction
    oss << (will_evict ? "[CONFLICT]" : "[COLD]");
    if (!is_prefetch) {
      oss << " acc=" << (stats_.demand_accesses + 1);
    }
    oss << " tile=" << la.tile;
    if (hout >= 0) {
      oss << " hout=" << hout << " wout=" << wout;
    }
    oss << " t=" << current_timestep_
        << " set=" << set_idx
        << " key=" << la.key
        << " cin=" << la.cin
        << " kh=" << la.kh
        << " kw=" << la.kw;
    WriteTrace(oss.str());
    // After a conflict miss, dump the full set state for debugging
    if (will_evict) {
      std::ostringstream osset;
      const int W = static_cast<int>(set.ways.size());
      osset << "[CacheSim][" << access_kind << "][SET]"
            << " set=" << set_idx
            << " ways=" << W;
      for (int i = 0; i < W; ++i) {
        const WayEntry& w = set.ways[static_cast<std::size_t>(i)];
        osset << " | way=" << i
              << " " << (w.valid ? "V" : "I")
              << " lru=" << w.lru_counter
              << " tag=" << w.tag
              << " cin=" << w.channel_id;
        if (w.valid) {
          // Derive full key depending on mapping mode
          const std::uint64_t full_key = cfg_.use_original_maptotag
              ? (w.tag * static_cast<std::uint64_t>(num_sets_) + static_cast<std::uint64_t>(set_idx))
              : w.tag;
          const std::uint32_t tile = static_cast<std::uint32_t>((full_key >> 40) & 0xFFFFFFull);
          const std::uint32_t kh   = static_cast<std::uint32_t>((full_key >> 12) & 0xFFFull);
          const std::uint32_t kw   = static_cast<std::uint32_t>( full_key        & 0xFFFull);
          osset << " tile=" << tile
                << " kh=" << kh
                << " kw=" << kw;
        }
      }
      WriteTrace(osset.str());
    }
  }

  // Count DM cold/conflict misses by (tile, output site, timestep)
  if (!is_prefetch && current_spine_id_ >= 0) {
    const int t = (current_timestep_ < 0) ? 0 : current_timestep_;
    auto& c = per_tile_site_step_counts_[static_cast<int>(la.tile)][current_spine_id_][t];
    if (will_evict) c.conflict_misses += 1ULL; else c.cold_misses += 1ULL;
  }

  if (victim_entry.valid && victim_entry.channel_id >= 0) {
    int score = 0;
    if (current_timestep_ > 0) {
      auto it_prev = scoreboard_by_t_.find(current_timestep_ - 1);
      if (it_prev != scoreboard_by_t_.end()) score = it_prev->second.Get(victim_entry.channel_id);
    }
    const uint64_t prev_key = victim_entry.tag; // tag stores full key under hashed indexing
    if (TraceHasCapacity()) {
      std::ostringstream oss;
      oss << "[CacheSim][" << access_kind << "][EVICT]";
      if (!is_prefetch) {
        oss << " acc=" << (stats_.demand_accesses + 1);
      }
      oss << " tile=" << la.tile;
      if (hout >= 0) {
        oss << " hout=" << hout << " wout=" << wout;
      }
      oss << " t=" << current_timestep_;

      // Compute durations since last access for the evicted line
      std::uint64_t cur_turn = access_sequence_counter_ + (is_prefetch ? 0ull : 1ull);
      long long since_turn = -1;
      auto it_turn = last_access_turn_.find(prev_key);
      if (it_turn != last_access_turn_.end()) {
        since_turn = static_cast<long long>((cur_turn >= it_turn->second) ? (cur_turn - it_turn->second) : 0ull);
      }
      int since_ts = -1;
      auto it_ts = last_access_timestep_.find(prev_key);
      if (it_ts != last_access_timestep_.end() && current_timestep_ >= 0) {
        since_ts = current_timestep_ - it_ts->second;
      }

      oss << " set=" << set_idx
          << " way=" << vic
          << " prev_key=" << prev_key
          << " prev_channel=" << victim_entry.channel_id
          << " score=" << score
          << " since_last_acc=" << since_turn;
      if (since_ts >= 0) {
        oss << " since_last_t=" << since_ts;
      }
      WriteTrace(oss.str());
    }
  }

  const int cost = cfg_.miss_overhead;

  // Install the line
  victim_entry.tag        = tag;
  victim_entry.valid      = true;
  victim_entry.channel_id = static_cast<int>(la.cin);
  TouchLRU(set, vic);

  // if (TraceHasCapacity()) {
  //   std::ostringstream oss;
  //   oss << "[CacheSim][" << access_kind << "][FILL]"
  //       << " set=" << set_idx
  //       << " way=" << vic
  //       << " key=" << la.key
  //       << " channel=" << la.cin;
  //   WriteTrace(oss.str());
  // }

  result.cycles = is_prefetch ? 0 : cost;
  result.miss = true;
  return result;
}

std::pair<int, uint64_t> CacheSim::MapToSetTag(uint64_t key) const {
  const uint64_t nsets = static_cast<uint64_t>(num_sets_);
  const int set_idx = static_cast<int>(key % nsets);
  const uint64_t tag = key / nsets;
  return { set_idx, tag };
}

std::pair<int, uint64_t> CacheSim::MapToSetTag(uint64_t key, int channel_id) const {
  const uint32_t nsets = static_cast<uint32_t>(num_sets_);
  const int set_idx = static_cast<int>(index_xorfold(key, nsets, static_cast<uint32_t>(channel_id)));
  // Use full key as tag to avoid dependence on index mapping.
  const uint64_t tag = key;
  return { set_idx, tag };
}

std::pair<int, uint64_t> CacheSim::MapToSetTag(uint64_t key, int channel_id, bool use_original) const {
  return use_original ? MapToSetTag(key) : MapToSetTag(key, channel_id);
}

int CacheSim::FindHit(Set& set, uint64_t tag) const {
  const int W = static_cast<int>(set.ways.size());
  for (int i = 0; i < W; ++i) {
    const WayEntry& w = set.ways[static_cast<std::size_t>(i)];
    if (w.valid && w.tag == tag) return i;
  }
  return -1;
}

void CacheSim::TouchLRU(Set& set, int way) {
  // Simple LRU aging: increment all, set touched way to 0.
  const int W = static_cast<int>(set.ways.size());
  for (int i = 0; i < W; ++i) {
    set.ways[static_cast<std::size_t>(i)].lru_counter++;
  }
  set.ways[static_cast<std::size_t>(way)].lru_counter = 0;
}

int CacheSim::PickVictim(int set_idx, Set& set, EvictionPolicy policy) {
  // Prefer an invalid way first
  const int W = static_cast<int>(set.ways.size());
  for (int i = 0; i < W; ++i) {
    if (!set.ways[static_cast<std::size_t>(i)].valid) {
      return i;
    }
  }

  switch (policy) {
    case EvictionPolicy::kScoreboard:
      // At t=0 we use LRU-only eviction regardless of scoreboard
      if (use_lru_this_step_) {
        return PickVictimLRU(set_idx, set);
      }
      return PickVictimScoreboard(set_idx, set);
    case EvictionPolicy::kLRU:
      return PickVictimLRU(set_idx, set);
    default:
      return PickVictimScoreboard(set_idx, set);
  }
}

int CacheSim::PickVictimScoreboard(int set_idx, Set& set) {
  const int W = static_cast<int>(set.ways.size());
  int min_score = INT_MAX;
  std::vector<int> candidates; candidates.reserve(W);
  const Scoreboard* prev_sb = nullptr;
  if (current_timestep_ > 0) {
    auto it_prev = scoreboard_by_t_.find(current_timestep_ - 1);
    if (it_prev != scoreboard_by_t_.end()) prev_sb = &it_prev->second;
  }
  for (int i = 0; i < W; ++i) {
    const WayEntry& w = set.ways[static_cast<std::size_t>(i)];
    const int sc = prev_sb ? prev_sb->Get(w.channel_id) : 0;
    if (sc < min_score) {
      if (sc == 0) { stats_.zero_score_events++; }
      min_score = sc;
      candidates.clear();
      candidates.push_back(i);
    } else if (sc == min_score) {
      candidates.push_back(i);
    }
  }

  int best = candidates.front();
  for (int idx : candidates) {
    if (set.ways[static_cast<std::size_t>(idx)].lru_counter >
        set.ways[static_cast<std::size_t>(best)].lru_counter) {
      best = idx;
    }
  }

  // // Trace the scoreboard decision in the same file as cache events
  // if (TraceHasCapacity()) {
  //   const WayEntry& chosen = set.ways[static_cast<std::size_t>(best)];
  //   std::ostringstream oss;
  //   oss << "[CacheSim][SB][CHOOSE]"
  //       << " set=" << set_idx
  //       << " way=" << best
  //       << " channel=" << chosen.channel_id
  //       << " score(S[t-1])=" << (prev_sb ? prev_sb->Get(chosen.channel_id) : 0)
  //       << " lru=" << chosen.lru_counter
  //       << " min_score=" << min_score
  //       << " cand_count=" << candidates.size() << '\n';
  //   // Dump the S[t-1] scoreboard used for this decision
  //   if (prev_sb) prev_sb->Dump(oss); else {
  //     Scoreboard empty;
  //     empty.Dump(oss);
  //   }
  //   WriteTrace(oss.str());
  // }
  return best;
}

int CacheSim::PickVictimLRU(int /*set_idx*/, Set& set) {
  const int W = static_cast<int>(set.ways.size());
  int best = 0;
  for (int i = 1; i < W; ++i) {
    if (set.ways[static_cast<std::size_t>(i)].lru_counter >
        set.ways[static_cast<std::size_t>(best)].lru_counter) {
      best = i;
    }
  }
  return best;
}

bool CacheSim::InSameTile(const LineAddr& a, const LineAddr& b) const {
  // Prefetch remains within the same tile and spatial position (kh,kw).
  return (a.tile == b.tile) && (a.kh == b.kh) && (a.kw == b.kw);
}

bool CacheSim::TraceHasCapacity() const {
  if (!cfg_.trace_enabled) {
    return false;
  }
  if (cfg_.trace_max_lines == 0) {
    return true;
  }
  return trace_lines_written_ < cfg_.trace_max_lines;
}

CacheSim::TileSiteStepMap CacheSim::TileSiteStepCountsSnapshot() const {
  return per_tile_site_step_counts_;
}

void CacheSim::WriteTrace(const std::string& message) {
  if (!TraceHasCapacity()) {
    return;
  }
  if (trace_stream_) {
    (*trace_stream_) << message << '\n';
    trace_stream_->flush();
  } else {
    std::cout << message << '\n';
  }
  trace_lines_written_++;
}



void PrintCacheConfig(const CacheConfig& cfg) {
  auto policy_to_string = [](EvictionPolicy policy) {
    switch (policy) {
      case EvictionPolicy::kScoreboard: return "scoreboard";
      case EvictionPolicy::kLRU:        return "lru";
      default:                          return "unknown";
    }
  };

  std::cout << "[CacheConfig] capacity_bytes=" << cfg.capacity_bytes
            << ", line_bytes=" << cfg.line_bytes
            << ", ways=" << cfg.ways
            << ", l1_hit_cycles=" << cfg.l1_hit_cycles
            << ", miss_overhead=" << cfg.miss_overhead
            << ", prefetch_depth=" << cfg.prefetch_depth
            << ", eviction_policy=" << policy_to_string(cfg.eviction_policy)
            << ", map=" << (cfg.use_original_maptotag ? "original" : "hashed")
            << '\n';
}

} // namespace sf::arch::cache
