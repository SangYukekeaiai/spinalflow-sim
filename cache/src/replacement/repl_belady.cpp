// All comments are in English.
#include "cache/belady.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cache/replacement_iface.h"

namespace sf::cache {

namespace {

class BeladyReplacement final : public IReplacement {
public:
  explicit BeladyReplacement(std::shared_ptr<const BeladyPlan> plan,
                             const CacheConfig& cfg)
      : plan_(std::move(plan)),
        expected_sets_(cfg.geometry.num_sets) {
    if (!plan_) {
      throw std::invalid_argument("BeladyReplacement: plan is null.");
    }
    if (plan_->num_sets != expected_sets_) {
      throw std::invalid_argument("BeladyReplacement: plan/set mismatch.");
    }
    if (expected_sets_ <= 0) {
      throw std::invalid_argument("BeladyReplacement: num_sets must be positive.");
    }
  }

  void InitSet(SetState& set, int ways) override {
    if (ways <= 0) {
      throw std::invalid_argument("BeladyReplacement::InitSet: ways must be positive.");
    }
    set.lines.resize(static_cast<std::size_t>(ways));
    const int set_idx = next_set_index_++;
    if (set_idx >= expected_sets_) {
      throw std::runtime_error("BeladyReplacement::InitSet: unexpected set index.");
    }
    set_index_[&set] = set_idx;
    auto& meta = next_use_by_way_[&set];
    meta.assign(static_cast<std::size_t>(ways), kBeladyNoFutureUse);
    cursor_ = 0;
    pending_ = {};
  }

  void ResetSet(SetState& set) override {
    for (auto& line : set.lines) {
      line = {};
    }
    auto& meta = next_use_by_way_[&set];
    std::fill(meta.begin(), meta.end(), kBeladyNoFutureUse);
    cursor_ = 0;
    pending_ = {};
  }

  int FindWay(const SetState& set, std::uint64_t tag) const override {
    auto it = set_index_.find(&set);
    if (it == set_index_.end()) {
      throw std::runtime_error("BeladyReplacement::FindWay: unknown set.");
    }
    const int set_idx = it->second;
    if (cursor_ >= plan_->accesses.size()) {
      throw std::runtime_error("BeladyReplacement::FindWay: plan exhausted.");
    }
    const auto& access = plan_->accesses[cursor_];
    if (access.set_idx != set_idx || access.tag != tag) {
      throw std::runtime_error("BeladyReplacement::FindWay: access mismatch with plan.");
    }
    pending_ = PendingAccess{&set, tag, access.next_use_index};
    pending_index_ = cursor_;
    cursor_ += 1;

    for (int way = 0; way < static_cast<int>(set.lines.size()); ++way) {
      const LineMeta& line = set.lines[static_cast<std::size_t>(way)];
      if (line.valid && line.tag == tag) {
        return way;
      }
    }
    return -1;
  }

  VictimInfo PickVictim(SetState& set) override {
    const int ways = static_cast<int>(set.lines.size());
    if (ways <= 0) {
      return {};
    }
    auto& meta = next_use_by_way_[&set];
    if (static_cast<int>(meta.size()) != ways) {
      meta.assign(static_cast<std::size_t>(ways), kBeladyNoFutureUse);
    }

    for (int way = 0; way < ways; ++way) {
      LineMeta& line = set.lines[static_cast<std::size_t>(way)];
      if (!line.valid) {
        meta[static_cast<std::size_t>(way)] = kBeladyNoFutureUse;
        return VictimInfo{way, false};
      }
    }

    int victim = 0;
    std::size_t farthest = meta.empty() ? kBeladyNoFutureUse : meta[0];
    for (int way = 1; way < ways; ++way) {
      const std::size_t candidate = meta[static_cast<std::size_t>(way)];
      if (candidate > farthest) {
        farthest = candidate;
        victim = way;
      }
    }
    meta[static_cast<std::size_t>(victim)] = kBeladyNoFutureUse;
    const bool valid = set.lines[static_cast<std::size_t>(victim)].valid;
    return VictimInfo{victim, valid};
  }

  void Install(SetState& set, int way, std::uint64_t tag, int tile_id) override {
    if (!pending_.set || pending_.set != &set) {
      throw std::runtime_error("BeladyReplacement::Install: missing pending access context.");
    }
    if (way < 0 || way >= static_cast<int>(set.lines.size())) {
      throw std::out_of_range("BeladyReplacement::Install: way out of range.");
    }
    LineMeta& line = set.lines[static_cast<std::size_t>(way)];
    line.valid = true;
    line.tag = tag;
    line.tile_id = tile_id;
    next_use_by_way_[&set][static_cast<std::size_t>(way)] = pending_.next_use;
    pending_ = {};
    pending_index_ = kBeladyNoFutureUse;
  }

  void OnHit(SetState& set, int way) override {
    if (!pending_.set || pending_.set != &set) {
      throw std::runtime_error("BeladyReplacement::OnHit: missing pending access context.");
    }
    if (way < 0 || way >= static_cast<int>(set.lines.size())) {
      throw std::out_of_range("BeladyReplacement::OnHit: way out of range.");
    }
    next_use_by_way_[&set][static_cast<std::size_t>(way)] = pending_.next_use;
    pending_ = {};
    pending_index_ = kBeladyNoFutureUse;
  }

  void OnPrefetchTouch(SetState& /*set*/, int /*way*/) override {
    // Prefetch is disabled for Belady; ignore touches.
  }

private:
  struct PendingAccess {
    const SetState* set = nullptr;
    std::uint64_t tag = 0;
    std::size_t next_use = kBeladyNoFutureUse;
  };

  std::shared_ptr<const BeladyPlan> plan_;
  int expected_sets_ = 0;
  std::unordered_map<const SetState*, int> set_index_;
  std::unordered_map<const SetState*, std::vector<std::size_t>> next_use_by_way_;

  mutable std::size_t cursor_ = 0;
  mutable PendingAccess pending_;
  mutable std::size_t pending_index_ = kBeladyNoFutureUse;
  int next_set_index_ = 0;
};

} // namespace

std::unique_ptr<IReplacement> MakeBeladyReplacement(const CacheConfig& cfg) {
  auto plan = GetBeladyPlan();
  if (!plan) {
    throw std::runtime_error("MakeBeladyReplacement: no Belady plan registered.");
  }
  return std::make_unique<BeladyReplacement>(std::move(plan), cfg);
}

} // namespace sf::cache
