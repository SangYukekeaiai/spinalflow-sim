// All comments are in English.
#include "cache/cache_iface.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include "cache/mapper_iface.h"
#include "cache/prefetch_iface.h"
#include "cache/replacement_iface.h"
#include "cache/replacement/temporal_iface.h"
#include "cache/window_iface.h"
#include "cache/registry.h"

namespace sf::cache {

class CacheCore final : public ICache {
public:
  CacheCore(CacheConfig cfg, CacheModules modules)
      : cfg_(std::move(cfg)),
        mapper_(std::move(modules.mapper)),
        replacement_(std::move(modules.replacement)),
        prefetch_(std::move(modules.prefetch)),
        window_(std::move(modules.window)) {
    cfg_.Validate();
    if (!mapper_ || !replacement_ || !window_) {
      throw std::invalid_argument("CacheCore: mapper, replacement, and window modules are required.");
    }
    const long long tile_prod =
        static_cast<long long>(cfg_.Cin) *
        static_cast<long long>(cfg_.KH) *
        static_cast<long long>(cfg_.KW);
    if (tile_prod <= 0 || tile_prod > std::numeric_limits<int>::max()) {
      throw std::invalid_argument("CacheCore: Cin * KH * KW overflow.");
    }
    const int N_color = cfg_.geometry.num_sets / 2;
    if (std::gcd(cfg_.A1, N_color) != 1) {
      throw std::invalid_argument("CacheCore: gcd(A1, num_sets/2) must be 1.");
    }
    sets_.resize(static_cast<std::size_t>(cfg_.geometry.num_sets));
    for (auto& set : sets_) {
      replacement_->InitSet(set, cfg_.geometry.ways);
    }
    temporal_policy_ = dynamic_cast<ITemporalReplacement*>(replacement_.get());
    Reset();
  }

  void Reset() override {
    for (auto& set : sets_) {
      replacement_->ResetSet(set);
    }
    stats_ = {};
    window_->Set(-1, -1);
    last_access_.reset();
    last_tile_ = -1;
    if (temporal_policy_) {
      temporal_policy_->OnTileStart(-1, -1);
    }
  }

  void SetWindow(int cur_tile, int next_tile) override {
    const int prev_tile = last_tile_;
    window_->Set(cur_tile, next_tile);
    if (temporal_policy_) {
      temporal_policy_->OnTileStart(cur_tile, prev_tile);
    }
    last_tile_ = cur_tile;
  }

  void AdvanceWindow() override {
    window_->Advance();
  }

  AccessResult OnDemandAccess(const AccessRequest& request) override {
    MapOutput map = mapper_->Map({request.tile_id, request.cin, request.kh, request.kw});
    if (map.set_idx < 0 || map.set_idx >= cfg_.geometry.num_sets) {
      throw std::out_of_range("CacheCore::OnDemandAccess: set index out of range.");
    }

    SetState& set = sets_[static_cast<std::size_t>(map.set_idx)];
    AccessResult result;
    result.tile_id = request.tile_id;
    result.set_idx = map.set_idx;
    result.L = map.L;

    const int way = replacement_->FindWay(set, map.tag);
    result.way = way;

    bool handled = false;

    if (way >= 0) {
      result.hit = true;
      result.latency_cycles = cfg_.timing.hit_latency_cycles;
      stats_.demand_hits += 1;
      stats_.latency_cycles += cfg_.timing.hit_latency_cycles;
      replacement_->OnHit(set, way);
      handled = true;
    } else if (temporal_policy_) {
      if (temporal_policy_->TryTemporalHit(request, map, result)) {
        stats_.demand_hits += 1;
        stats_.latency_cycles += cfg_.timing.hit_latency_cycles;
        handled = true;
      } else if (request.tile_id == window_->Cur()) {
        if (temporal_policy_->HandleCurTileMiss(set, request, map, result)) {
          stats_.demand_misses_allocated += 1;
          stats_.demand_bytes_loaded += static_cast<std::uint64_t>(cfg_.geometry.line_size_bytes);
          stats_.latency_cycles += cfg_.timing.miss_latency_cycles;
          handled = true;
        }
      }
    }

    if (!handled) {
      const bool window_ok = window_->Allows(request.tile_id);
      result.window_admitted = window_ok;
      result.latency_cycles = cfg_.timing.miss_latency_cycles;
      stats_.latency_cycles += cfg_.timing.miss_latency_cycles;
      if (!window_ok || request.tile_id == window_->Next()) {
        stats_.demand_misses_noalloc += 1;
      } else {
        stats_.demand_misses_allocated += 1;
        VictimInfo victim = replacement_->PickVictim(set);
        if (victim.way < 0 || victim.way >= cfg_.geometry.ways) {
          throw std::runtime_error("CacheCore::OnDemandAccess: invalid victim way.");
        }
        if (victim.was_valid) {
          stats_.evictions_total += 1;
        }
        replacement_->Install(set, victim.way, map.tag, request.tile_id);
        result.bytes_fetched = static_cast<std::uint64_t>(cfg_.geometry.line_size_bytes);
        stats_.demand_bytes_loaded += result.bytes_fetched;
        result.allocated = true;
        result.way = victim.way;
      }
    }

    last_access_ = result;

    if (prefetch_) {
      PrefetchInput input;
      input.cur_tile = window_->Cur();
      input.next_tile = window_->Next();
      input.map = map;
      input.request = request;
      PrefetchPlan plan = prefetch_->Plan(input);
      if (plan.do_prefetch &&
          plan.tile_id >= 0 &&
          plan.target_map.set_idx >= 0 &&
          plan.target_map.set_idx < cfg_.geometry.num_sets) {
        SetState& prefetch_set = sets_[static_cast<std::size_t>(plan.target_map.set_idx)];
        const int prefetch_way = replacement_->FindWay(prefetch_set, plan.target_map.tag);
        if (prefetch_way >= 0) {
          stats_.prefetch_hits += 1;
          replacement_->OnPrefetchTouch(prefetch_set, prefetch_way);
        } else {
          VictimInfo victim = replacement_->PickVictim(prefetch_set);
          if (victim.way < 0 || victim.way >= cfg_.geometry.ways) {
            throw std::runtime_error("CacheCore::OnDemandAccess: invalid prefetch victim.");
          }
          if (victim.was_valid) {
            stats_.evictions_total += 1;
          }
          replacement_->Install(prefetch_set, victim.way, plan.target_map.tag, plan.tile_id);
          stats_.prefetch_inserts += 1;
          stats_.prefetch_bytes_loaded += static_cast<std::uint64_t>(cfg_.geometry.line_size_bytes);
        }
      }
    }

    return result;
  }

  const CacheStats& Stats() const override {
    return stats_;
  }

private:
  CacheConfig cfg_;
  std::unique_ptr<IMapper> mapper_;
  std::unique_ptr<IReplacement> replacement_;
  ITemporalReplacement* temporal_policy_ = nullptr;
  std::unique_ptr<IPrefetch> prefetch_;
  std::unique_ptr<IWindow> window_;

  std::vector<SetState> sets_;
  CacheStats stats_{};
  std::optional<AccessResult> last_access_;
  int last_tile_ = -1;
};

std::unique_ptr<ICache> BuildCache(const CacheConfig& cfg,
                                   CacheModules modules) {
  return std::make_unique<CacheCore>(cfg, std::move(modules));
}

} // namespace sf::cache
