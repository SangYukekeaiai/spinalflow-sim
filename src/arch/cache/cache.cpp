// All comments are in English.

#include "arch/cache/cache.hpp"

#include <cmath>
#include <climits>
#include <iostream>
#include <filesystem>
#include <sstream>
#include <algorithm>
#include <stdexcept>

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
  // Emit all per-timestep snapshots at the end of the layer
  for (const auto& kv : scoreboard_by_t_) {
    EmitStepSnapshotIfEnabled_(kv.first, kv.second.Snapshot());
  }
  scoreboard_by_t_.clear();
  scoreboard_total_.Clear();
  current_timestep_ = -1;
  tmpZeroScoreCount_ = 0;
  unique_demand_lines_seen_.clear();
  last_access_turn_.clear();
  access_sequence_counter_ = 0;
  stats_ = {};
  if (trace_stream_) {
    trace_stream_->flush();
  }
}

void CacheSim::NotifySpike(int cin) {
  const int t = (current_timestep_ >= 0) ? current_timestep_ : 0;
  scoreboard_by_t_[t].Bump(cin);
  scoreboard_total_.Bump(cin);
}

void CacheSim::BeginTimeStep(int t) {
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
    auto [uc_set_idx, tag_unused] = MapToSetTag(la.key, static_cast<int>(la.cin));
    (void)tag_unused;
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
    last_it->second = current_turn;
  } else {
    last_it->second = current_turn;
  }
  if (demand.miss) {
    // std::cout << "Miss--Try to load address: " << la.key
    //           << " (tile=" << la.tile << ", cin=" << la.cin
    //           << ", kh=" << la.kh << ", kw=" << la.kw << ")\n";
    stats_.demand_misses++;
    stats_.demand_miss_cycles += static_cast<std::uint64_t>(demand.cycles);
  } else {
    stats_.demand_hit_cycles += static_cast<std::uint64_t>(demand.cycles);
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
  auto [set_idx, tag] = MapToSetTag(la.key, static_cast<int>(la.cin));
  const char* access_kind = is_prefetch ? "PF" : "DM";
  Set& set = sets_[static_cast<std::size_t>(set_idx)];

  // Check hit
  const int hit_way = FindHit(set, tag);
  if (hit_way >= 0) {
    if (TraceHasCapacity()) {
      std::ostringstream oss;
      oss << "[CacheSim][" << access_kind << "][HIT]"
          << " set=" << set_idx
          << " way=" << hit_way
          << " key=" << la.key
          << " tile=" << la.tile
          << " cin=" << la.cin
          << " kh=" << la.kh
          << " kw=" << la.kw;
      WriteTrace(oss.str());
    }
    TouchLRU(set, hit_way);
    result.cycles = is_prefetch ? 0 : cfg_.l1_hit_cycles;
    result.miss = false;
    return result;
  }

  if (TraceHasCapacity()) {
    std::ostringstream oss;
    oss << "[CacheSim][" << access_kind << "][MISS]"
        << " set=" << set_idx
        << " key=" << la.key
        << " tile=" << la.tile
        << " cin=" << la.cin
        << " kh=" << la.kh
        << " kw=" << la.kw;
    WriteTrace(oss.str());

    // Dump all valid lines currently present in this set
    std::ostringstream voss;
    const int W = static_cast<int>(set.ways.size());
    voss << "[CacheSim][" << access_kind << "][SET_VALID]"
         << " set=" << set_idx;
    bool first = true;
    for (int i = 0; i < W; ++i) {
      const WayEntry& w = set.ways[static_cast<std::size_t>(i)];
      if (!w.valid) continue;
      const uint64_t line_key = w.tag; // tag stores full key under hashed indexing
      voss << (first ? " lines=" : ", ");
      voss << "(way=" << i
           << " key=" << line_key
           << " channel=" << w.channel_id
           << " lru=" << w.lru_counter
           << " score(S[t-1])=" << ((current_timestep_ > 0) ? scoreboard_by_t_[current_timestep_ - 1].Get(w.channel_id) : 0)
           << ")";
      first = false;
    }
    WriteTrace(voss.str());
  }

  const int vic = PickVictim(set_idx, set, policy);
  WayEntry& victim_entry = set.ways[static_cast<std::size_t>(vic)];
  if (victim_entry.valid && victim_entry.channel_id >= 0) {
    const int score = (current_timestep_ > 0) ? scoreboard_by_t_[current_timestep_ - 1].Get(victim_entry.channel_id) : 0;
    const uint64_t prev_key = victim_entry.tag; // tag stores full key under hashed indexing
    if (TraceHasCapacity()) {
      std::ostringstream oss;
      oss << "[CacheSim][" << access_kind << "][EVICT]"
          << " set=" << set_idx
          << " way=" << vic
          << " prev_key=" << prev_key
          << " prev_channel=" << victim_entry.channel_id
          << " score=" << score;
      WriteTrace(oss.str());
    }
  }

  const int cost = cfg_.miss_overhead;

  // Install the line
  victim_entry.tag        = tag;
  victim_entry.valid      = true;
  victim_entry.channel_id = static_cast<int>(la.cin);
  TouchLRU(set, vic);

  if (TraceHasCapacity()) {
    std::ostringstream oss;
    oss << "[CacheSim][" << access_kind << "][FILL]"
        << " set=" << set_idx
        << " way=" << vic
        << " key=" << la.key
        << " channel=" << la.cin;
    WriteTrace(oss.str());
  }

  result.cycles = is_prefetch ? 0 : cost;
  result.miss = true;
  return result;
}

std::pair<int, uint64_t> CacheSim::MapToSetTag(uint64_t key, int channel_id) const {
  const uint32_t nsets = static_cast<uint32_t>(num_sets_);
  const int set_idx = static_cast<int>(index_xorfold(key, nsets, static_cast<uint32_t>(channel_id)));
  // Use full key as tag to avoid dependence on index mapping.
  const uint64_t tag = key;
  return { set_idx, tag };
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
      if (current_timestep_ <= 0) {
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
  for (int i = 0; i < W; ++i) {
    const WayEntry& w = set.ways[static_cast<std::size_t>(i)];
    const int sc = (current_timestep_ > 0) ? scoreboard_by_t_[current_timestep_ - 1].Get(w.channel_id) : 0;
    if (sc < min_score) {
      if (sc == 0) {
        tmpZeroScoreCount_++;
        stats_.zero_score_events++;
      }
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

  // Trace the scoreboard decision in the same file as cache events
  if (TraceHasCapacity()) {
    const WayEntry& chosen = set.ways[static_cast<std::size_t>(best)];
    std::ostringstream oss;
    oss << "[CacheSim][SB][CHOOSE]"
        << " set=" << set_idx
        << " way=" << best
        << " channel=" << chosen.channel_id
        << " score(S[t-1])=" << ((current_timestep_ > 0) ? scoreboard_by_t_[current_timestep_ - 1].Get(chosen.channel_id) : 0)
        << " lru=" << chosen.lru_counter
        << " min_score=" << min_score
        << " cand_count=" << candidates.size() << '\n';
    // Dump the S[t-1] scoreboard used for this decision
    if (current_timestep_ > 0) {
      scoreboard_by_t_[current_timestep_ - 1].Dump(oss);
    } else {
      oss << "[Scoreboard] empty\n";
    }
    WriteTrace(oss.str());
  }
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

bool CacheSim::TraceAvailable() const {
  return static_cast<bool>(trace_stream_);
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

// -------------------- Per-step scoreboard CSV emission -----------------------

void CacheSim::EmitStepSnapshotIfEnabled_(int t,
                                          const std::unordered_map<int,int>& scores) {
  if (!cfg_.trace_enabled || cfg_.trace_output_path.empty()) {
    return; // respect existing trace toggle as a proxy for step snapshots
  }
  namespace fs = std::filesystem;
  try {
    const std::string csv_path_str = BuildStepCsvPath_(t);
    if (csv_path_str.empty()) return;
    const fs::path csv_path(csv_path_str);
    fs::create_directories(csv_path.parent_path());
    std::ofstream ofs(csv_path, std::ios::out | std::ios::trunc);
    if (!ofs) {
      return; // silently skip on failure to avoid disrupting sim
    }
    // Build map: score -> vector of channels
    std::unordered_map<int, std::vector<int>> by_score;
    by_score.reserve(scores.size());
    for (const auto& kv : scores) {
      by_score[kv.second].push_back(kv.first);
    }
    for (auto& kv : by_score) {
      auto& chans = kv.second;
      std::sort(chans.begin(), chans.end());
    }
    std::vector<int> svals; svals.reserve(by_score.size());
    for (const auto& kv : by_score) svals.push_back(kv.first);
    std::sort(svals.begin(), svals.end());

    ofs << "score,num_channels,channel_ids\n";
    for (int sc : svals) {
      const auto& chans = by_score.at(sc);
      ofs << sc << ',' << chans.size() << ',';
      ofs << '"';
      for (std::size_t i = 0; i < chans.size(); ++i) {
        ofs << chans[i];
        if (i + 1 < chans.size()) ofs << ", ";
      }
      ofs << '"' << '\n';
    }
    ofs.flush();
  } catch (...) {
    // swallow to keep cache sim robust
  }
}

std::string CacheSim::BuildStepCsvPath_(int t) const {
  if (cfg_.trace_output_path.empty()) return std::string();
  namespace fs = std::filesystem;
  const fs::path trace_path(cfg_.trace_output_path);
  // trace path example:
  //   .../layer5/cache_traces/<policy>/<ways>_<prefetch>/<size>.txt
  // Produce per-step scoreboard under:
  //   .../layer5/scoreboard_steps/<policy>/<ways>_<prefetch>/
  //   scoreboard_scores_<size>KB_<ways>_<prefetch>_<policy>_t<t>.csv
  fs::path ways_prefetch_dir = trace_path.parent_path();             // <ways>_<prefetch>
  fs::path policy_dir        = ways_prefetch_dir.parent_path();      // <policy>
  fs::path cache_traces_dir  = policy_dir.parent_path();             // cache_traces
  fs::path layer_dir         = cache_traces_dir.parent_path();       // layerX
  const std::string policy_tag = policy_dir.filename().string();
  const std::string ways_prefetch = ways_prefetch_dir.filename().string();

  // Derive size_kb from capacity_bytes
  const std::size_t size_kb = cfg_.capacity_bytes / 1024u;
  // ways_prefetch is already e.g., "16_0"; keep consistent with existing
  fs::path steps_dir = layer_dir / "scoreboard_steps" / policy_tag / ways_prefetch;
  const std::string fname =
      std::string("scoreboard_scores_") + std::to_string(size_kb) + "KB_" +
      ways_prefetch + "_" + policy_tag + "_t" + std::to_string(t) + ".csv";
  return (steps_dir / fname).string();
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
            << '\n';
}

} // namespace sf::arch::cache
