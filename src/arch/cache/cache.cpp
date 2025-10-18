// All comments are in English.

#include "arch/cache/cache.hpp"

#include <cmath>
#include <climits>
#include <iostream>
#include <filesystem>
#include <sstream>
#include <stdexcept>

namespace sf::arch::cache {

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
  bool first = true;
  for (const auto& [channel, score] : scores_) {
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
  scoreboard_.Clear();
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
  scoreboard_.Bump(cin);
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
  auto [set_idx, tag] = MapToSetTag(la.key);
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
  }

  const int vic = PickVictim(set, policy);
  WayEntry& victim_entry = set.ways[static_cast<std::size_t>(vic)];
  if (victim_entry.valid && victim_entry.channel_id >= 0) {
    const int score = scoreboard_.Get(victim_entry.channel_id);
    const uint64_t prev_key =
        victim_entry.tag * static_cast<uint64_t>(num_sets_) +
        static_cast<uint64_t>(set_idx);
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

std::pair<int, uint64_t> CacheSim::MapToSetTag(uint64_t key) const {
  const uint64_t nsets = static_cast<uint64_t>(num_sets_);
  const int set_idx = static_cast<int>(key % nsets);
  const uint64_t tag = key / nsets;
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

int CacheSim::PickVictim(Set& set, EvictionPolicy policy) {
  // Prefer an invalid way first
  const int W = static_cast<int>(set.ways.size());
  for (int i = 0; i < W; ++i) {
    if (!set.ways[static_cast<std::size_t>(i)].valid) {
      return i;
    }
  }

  switch (policy) {
    case EvictionPolicy::kScoreboard:
      return PickVictimScoreboard(set);
    case EvictionPolicy::kLRU:
      return PickVictimLRU(set);
    default:
      return PickVictimScoreboard(set);
  }
}

int CacheSim::PickVictimScoreboard(Set& set) {
  const int W = static_cast<int>(set.ways.size());
  int min_score = INT_MAX;
  std::vector<int> candidates; candidates.reserve(W);
  for (int i = 0; i < W; ++i) {
    const WayEntry& w = set.ways[static_cast<std::size_t>(i)];
    const int sc = scoreboard_.Get(w.channel_id);
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
  return best;
}

int CacheSim::PickVictimLRU(Set& set) {
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
