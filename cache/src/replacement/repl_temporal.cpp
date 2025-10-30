// All comments are in English.
#include "cache/replacement/temporal_iface.h"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <unordered_map>

#include "cache/cache_config.h"

namespace sf::cache {

namespace {
constexpr int kTemporalCapacity = 512;
constexpr int kRankBits = 3;
constexpr int kRankMax = (1 << kRankBits) - 1;
constexpr int kRankInitial = 1;
constexpr int kRankPromotionThreshold = 2;
constexpr int kRankDelta = 1;

inline std::uint8_t ClampRank(int value) {
  if (value < 0) return 0;
  if (value > kRankMax) return static_cast<std::uint8_t>(kRankMax);
  return static_cast<std::uint8_t>(value);
}

inline std::uint8_t SaturatingBump(std::uint8_t rank) {
  if (rank >= kRankMax) return static_cast<std::uint8_t>(kRankMax);
  return static_cast<std::uint8_t>(rank + 1);
}

inline int PositiveMod(long long value, int mod) {
  if (mod <= 0) return 0;
  long long res = value % mod;
  if (res < 0) res += mod;
  return static_cast<int>(res);
}

} // namespace

class TemporalAwareReplacement final : public IReplacement, public ITemporalReplacement {
public:
  explicit TemporalAwareReplacement(const CacheConfig& cfg)
      : cfg_(cfg),
        S_tile_(cfg.Cin * cfg.KH * cfg.KW),
        N_color_(cfg.geometry.num_sets / 2) {
    if (S_tile_ <= 0) {
      throw std::invalid_argument("TemporalAwareReplacement: invalid tile size.");
    }
    if (N_color_ <= 0) {
      throw std::invalid_argument("TemporalAwareReplacement: num_sets/2 must be positive.");
    }
    set_ptrs_.reserve(static_cast<std::size_t>(cfg.geometry.num_sets));
  }

  // IReplacement implementation -------------------------------------------------
  void InitSet(SetState& set, int ways) override {
    set.lines.resize(static_cast<std::size_t>(ways));
    set.probation_order.resize(static_cast<std::size_t>(ways));
    for (int i = 0; i < ways; ++i) {
      set.probation_order[static_cast<std::size_t>(i)] = i;
    }
    set.protected_order.clear();

    const int idx = static_cast<int>(set_ptrs_.size());
    set_ptrs_.push_back(&set);
    set_index_[&set] = idx;
  }

  void ResetSet(SetState& set) override {
    for (auto& line : set.lines) {
      line = {};
    }
    for (std::size_t i = 0; i < set.probation_order.size(); ++i) {
      set.probation_order[i] = static_cast<int>(i);
    }
    set.protected_order.clear();
  }

  int FindWay(const SetState& set, std::uint64_t tag) const override {
    for (std::size_t i = 0; i < set.lines.size(); ++i) {
      if (set.lines[i].valid && set.lines[i].tag == tag) {
        return static_cast<int>(i);
      }
    }
    return -1;
  }

  VictimInfo PickVictim(SetState& set) override {
    VictimInfo victim;
    victim.way = PickSetVictim(set);
    if (victim.way < 0) {
      return victim;
    }
    const auto& line = set.lines[static_cast<std::size_t>(victim.way)];
    victim.was_valid = line.valid;
    victim.residency = line.residency;
    return victim;
  }

  void Install(SetState& set, int way, std::uint64_t tag, int tile_id) override {
    auto& line = set.lines[static_cast<std::size_t>(way)];
    line.valid = true;
    line.tag = tag;
    line.tile_id = tile_id;
    line.touches = kRankInitial;
    TouchSet(set, way);
  }

  void OnHit(SetState& set, int way) override {
    auto& line = set.lines[static_cast<std::size_t>(way)];
    if (!line.valid) return;
    line.touches = SaturatingBump(static_cast<std::uint8_t>(line.touches));
    TouchSet(set, way);
  }

  void OnPrefetchTouch(SetState& set, int way) override {
    OnHit(set, way);
  }

  // ITemporalReplacement implementation -----------------------------------------
  void OnTileStart(int new_tile, int prev_tile) override {
    cur_tile_ = new_tile;

    ClearTemporal();
    if (prev_tile < 0) {
      return;
    }
    for (auto* set : set_ptrs_) {
      if (!set) continue;
      bool touched = false;
      for (auto& line : set->lines) {
        if (line.valid && line.tile_id == prev_tile) {
          line = {};
          touched = true;
        }
      }
      if (touched) {
        for (std::size_t i = 0; i < set->probation_order.size(); ++i) {
          set->probation_order[i] = static_cast<int>(i);
        }
      }
    }
  }

  bool TryTemporalHit(const AccessRequest& /*request*/,
                      const MapOutput& map,
                      AccessResult& result) override {
    const int idx = FindTemporal(map.tag);
    if (idx < 0) {
      return false;
    }
    auto& entry = temporal_[idx];
    if (!entry.valid || entry.tile_id != cur_tile_) {
      return false;
    }
    entry.rank = SaturatingBump(entry.rank);
    entry.lru = ++temp_timestamp_;

    result.hit = true;
    result.allocated = false;
    result.window_admitted = true;
    result.latency_cycles = cfg_.timing.hit_latency_cycles;
    result.bytes_fetched = 0;
    PromoteIfNeeded(false);
    return true;
  }

  bool HandleCurTileMiss(SetState& /*set*/,
                         const AccessRequest& request,
                         const MapOutput& map,
                         AccessResult& result) override {
    if (cur_tile_ < 0 || request.tile_id != cur_tile_) {
      return false;
    }

    result.hit = false;
    result.allocated = true;
    result.window_admitted = true;
    result.bytes_fetched = static_cast<std::uint64_t>(cfg_.geometry.line_size_bytes);
    result.latency_cycles = cfg_.timing.miss_latency_cycles;

    int idx = FindTemporal(map.tag);
    if (idx >= 0) {
      TouchTemporal(idx);
      PromoteIfNeeded(false);
      return true;
    }

    if (temp_count_ == kTemporalCapacity) {
      PromoteIfNeeded(true);
      if (temp_count_ == kTemporalCapacity) {
        EvictTemporalLowest();
      }
    }

    const int slot = FindTemporalFreeSlot();
    if (slot < 0) {
      return true; // nothing else we can do; treat as handled
    }
    auto& entry = temporal_[slot];
    entry.valid = true;
    entry.tag = map.tag;
    entry.tile_id = request.tile_id;
    entry.L = map.L;
    entry.rank = kRankInitial;
    entry.lru = ++temp_timestamp_;
    temp_count_ += 1;

    PromoteIfNeeded(false);
    return true;
  }

private:
  struct TempEntry {
    bool valid = false;
    std::uint64_t tag = 0;
    int tile_id = -1;
    int L = -1;
    std::uint8_t rank = 0;
    std::uint64_t lru = 0;
  };

  CacheConfig cfg_;
  int S_tile_ = 0;
  int N_color_ = 0;
  int cur_tile_ = -1;

  std::vector<SetState*> set_ptrs_;
  std::unordered_map<const SetState*, int> set_index_;

  std::array<TempEntry, kTemporalCapacity> temporal_{};
  int temp_count_ = 0;
  std::uint64_t temp_timestamp_ = 0;

  int FindTemporal(std::uint64_t tag) const {
    for (int i = 0; i < kTemporalCapacity; ++i) {
      if (temporal_[i].valid && temporal_[i].tag == tag) {
        return i;
      }
    }
    return -1;
  }

  int FindTemporalFreeSlot() const {
    for (int i = 0; i < kTemporalCapacity; ++i) {
      if (!temporal_[i].valid) {
        return i;
      }
    }
    return -1;
  }

  void TouchTemporal(int idx) {
    if (idx < 0 || idx >= kTemporalCapacity) return;
    auto& entry = temporal_[idx];
    if (!entry.valid) return;
    entry.rank = SaturatingBump(entry.rank);
    entry.lru = ++temp_timestamp_;
  }

  void ClearTemporal() {
    for (auto& entry : temporal_) {
      entry = {};
    }
    temp_count_ = 0;
    temp_timestamp_ = 0;
  }

  int ComputeSetIdx(int tile_id, int L) const {
    const int idx_in_color = PositiveMod(static_cast<long long>(cfg_.A1) * static_cast<long long>(L), N_color_);
    const int parity = tile_id & 1;
    return (idx_in_color << 1) | parity;
  }

  SetState* SetForIndex(int set_idx) const {
    if (set_idx < 0 || set_idx >= static_cast<int>(set_ptrs_.size())) {
      return nullptr;
    }
    return set_ptrs_[static_cast<std::size_t>(set_idx)];
  }

  void PromoteIfNeeded(bool force) {
    int candidate = -1;
    for (int i = 0; i < kTemporalCapacity; ++i) {
      const auto& entry = temporal_[i];
      if (!entry.valid) continue;
      if (!force && entry.rank < kRankPromotionThreshold) continue;
      if (candidate < 0) {
        candidate = i;
      } else {
        const auto& best = temporal_[candidate];
        if (entry.rank > best.rank || (entry.rank == best.rank && entry.lru < best.lru)) {
          candidate = i;
        }
      }
    }

    if (candidate < 0) {
      return;
    }

    if (PromoteEntry(candidate)) {
      return;
    }

    if (force) {
      EvictTemporalLowest();
    }
  }

  bool PromoteEntry(int idx) {
    if (idx < 0 || idx >= kTemporalCapacity) {
      return false;
    }
    const auto entry = temporal_[idx];
    if (!entry.valid) {
      return false;
    }

    const int set_idx = ComputeSetIdx(entry.tile_id, entry.L);
    SetState* set = SetForIndex(set_idx);
    if (!set) {
      return false;
    }

    const int existing = FindWay(*set, entry.tag);
    if (existing >= 0) {
      auto& line = set->lines[static_cast<std::size_t>(existing)];
      line.touches = ClampRank(line.touches + entry.rank);
      TouchSet(*set, existing);
      RemoveTemporal(idx);
      return true;
    }

    int victim = PickSetVictim(*set);
    if (victim < 0) {
      return false;
    }
    auto& line = set->lines[static_cast<std::size_t>(victim)];
    if (line.valid) {
      if (entry.rank + 0 < line.touches + kRankDelta) {
        return false;
      }
    }

    line.valid = true;
    line.tag = entry.tag;
    line.tile_id = entry.tile_id;
    line.touches = entry.rank;
    TouchSet(*set, victim);
    RemoveTemporal(idx);
    return true;
  }

  void RemoveTemporal(int idx) {
    if (idx < 0 || idx >= kTemporalCapacity) return;
    if (!temporal_[idx].valid) return;
    temporal_[idx] = {};
    temp_count_ -= 1;
    if (temp_count_ < 0) temp_count_ = 0;
  }

  void EvictTemporalLowest() {
    int victim = -1;
    for (int i = 0; i < kTemporalCapacity; ++i) {
      const auto& entry = temporal_[i];
      if (!entry.valid) continue;
      if (victim < 0) {
        victim = i;
        continue;
      }
      const auto& best = temporal_[victim];
      if (entry.rank < best.rank || (entry.rank == best.rank && entry.lru < best.lru)) {
        victim = i;
      }
    }
    if (victim >= 0) {
      RemoveTemporal(victim);
    }
  }

  int PickSetVictim(SetState& set) const {
    for (std::size_t i = 0; i < set.lines.size(); ++i) {
      if (!set.lines[i].valid) {
        return static_cast<int>(i);
      }
    }

    int victim = -1;
    int victim_rank = std::numeric_limits<int>::max();
    int victim_lru_pos = -1;

    for (std::size_t pos = 0; pos < set.probation_order.size(); ++pos) {
      const int way = set.probation_order[pos];
      const auto& line = set.lines[static_cast<std::size_t>(way)];
      if (line.touches < victim_rank) {
        victim = way;
        victim_rank = line.touches;
        victim_lru_pos = static_cast<int>(pos);
      }
    }

    if (victim < 0) {
      return -1;
    }

    for (std::size_t pos = 0; pos < set.probation_order.size(); ++pos) {
      const int way = set.probation_order[pos];
      const auto& line = set.lines[static_cast<std::size_t>(way)];
      if (!line.valid) continue;
      if (line.touches == victim_rank && static_cast<int>(pos) > victim_lru_pos) {
        victim = way;
        victim_lru_pos = static_cast<int>(pos);
      }
    }
    return victim;
  }

  void TouchSet(SetState& set, int way) {
    auto& order = set.probation_order;
    auto it = std::find(order.begin(), order.end(), way);
    if (it != order.end()) {
      order.erase(it);
    }
    order.insert(order.begin(), way);
  }
};

std::unique_ptr<IReplacement> MakeTemporalAwareReplacement(const CacheConfig& cfg) {
  return std::make_unique<TemporalAwareReplacement>(cfg);
}

} // namespace sf::cache
