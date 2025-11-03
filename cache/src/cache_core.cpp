// All comments are in English.
#include "cache/cache_iface.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "cache/mapper_iface.h"
#include "cache/prefetch_iface.h"
#include "cache/replacement_iface.h"
#include "cache/window_iface.h"
#include "cache/cache_hooks.h"
#include "cache/registry.h"
#include "cache/prefetch_buffer.h"

namespace sf::cache {

class CacheCore final : public ICache {
public:
  CacheCore(CacheConfig cfg, CacheModules modules)
      : cfg_(std::move(cfg)),
        mapper_(std::move(modules.mapper)),
        replacement_(std::move(modules.replacement)),
        prefetch_(std::move(modules.prefetch)),
        window_(std::move(modules.window)),
        prefetch_buffer_(std::move(modules.prefetch_buffer)) {
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
    sets_.resize(static_cast<std::size_t>(cfg_.geometry.num_sets));
    for (auto& set : sets_) {
      replacement_->InitSet(set, cfg_.geometry.ways);
    }
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
    if (prefetch_buffer_) {
      prefetch_buffer_->Clear();
    }
  }

  void SetWindow(int cur_tile, int next_tile) override {
    window_->Set(cur_tile, next_tile);
    if (prefetch_buffer_ && cur_tile >= 0) {
      LoadPrefetchBufferForTile(cur_tile);
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
    result.tag = map.tag;
    result.evicted = false;
    result.evicted_tag = 0;
    result.evicted_tile_id = -1;
    result.evicted_output_spine_id = -1;
    result.evicted_last_timestep = -1;

    const int way = replacement_->FindWay(set, map.tag);
    result.way = way;

    if (way >= 0) {
      result.hit = true;
      result.latency_cycles = cfg_.timing.hit_latency_cycles;
      stats_.demand_hits += 1;
      stats_.latency_cycles += cfg_.timing.hit_latency_cycles;
      replacement_->OnHit(set, way);
      if (way >= 0 && way < static_cast<int>(set.lines.size())) {
        LineMeta& line = set.lines[static_cast<std::size_t>(way)];
        line.output_spine_id = request.output_spine_id;
        line.last_timestep = request.timestep;
      }
    } else {
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
        LineMeta& victim_line = set.lines[static_cast<std::size_t>(victim.way)];
        if (victim.was_valid) {
          stats_.evictions_total += 1;
          result.evicted = true;
          result.evicted_tag = victim_line.tag;
          result.evicted_tile_id = victim_line.tile_id;
          result.evicted_output_spine_id = victim_line.output_spine_id;
          result.evicted_last_timestep = victim_line.last_timestep;
        }
        replacement_->Install(set, victim.way, map.tag, request.tile_id);
        LineMeta& new_line = set.lines[static_cast<std::size_t>(victim.way)];
        new_line.output_spine_id = request.output_spine_id;
        new_line.last_timestep = request.timestep;
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
      if (plan.do_prefetch && plan.request.tile_id >= 0) {
        MapInput target_input;
        target_input.tile_id = plan.request.tile_id;
        target_input.cin = plan.request.cin;
        target_input.kh = plan.request.kh;
        target_input.kw = plan.request.kw;
        MapOutput target_map = mapper_->Map(target_input);
        if (target_map.set_idx >= 0 && target_map.set_idx < cfg_.geometry.num_sets) {
          bool stored_in_buffer = false;
          bool skip_prefetch = false;
          if (prefetch_buffer_) {
            if (!prefetch_buffer_->Store(plan.request.tile_id, target_map)) {
              skip_prefetch = true;
            } else {
              stored_in_buffer = true;
            }
          }
          if (!skip_prefetch && !stored_in_buffer) {
            SetState& prefetch_set = sets_[static_cast<std::size_t>(target_map.set_idx)];
            const int prefetch_way = replacement_->FindWay(prefetch_set, target_map.tag);
            if (prefetch_way >= 0) {
              stats_.prefetch_hits += 1;
              replacement_->OnPrefetchTouch(prefetch_set, prefetch_way);
              if (prefetch_way >= 0 && prefetch_way < static_cast<int>(prefetch_set.lines.size())) {
                LineMeta& line = prefetch_set.lines[static_cast<std::size_t>(prefetch_way)];
                line.last_timestep = -1;
                line.output_spine_id = -1;
              }
            } else {
              VictimInfo victim = replacement_->PickVictim(prefetch_set);
              if (victim.way < 0 || victim.way >= cfg_.geometry.ways) {
                throw std::runtime_error("CacheCore::OnDemandAccess: invalid prefetch victim.");
              }
              if (victim.was_valid) {
                stats_.evictions_total += 1;
              }
              replacement_->Install(prefetch_set, victim.way, target_map.tag, plan.request.tile_id);
              LineMeta& prefetch_line = prefetch_set.lines[static_cast<std::size_t>(victim.way)];
              prefetch_line.output_spine_id = -1;
              prefetch_line.last_timestep = -1;
              stats_.prefetch_inserts += 1;
              stats_.prefetch_bytes_loaded += static_cast<std::uint64_t>(cfg_.geometry.line_size_bytes);
            }
          }
        }
      }
    }

    return result;
  }

  void LoadPrefetchBufferForTile(int tile_id) {
    auto entries = prefetch_buffer_->TakeForTile(tile_id);
    for (const auto& entry : entries) {
      if (entry.map.set_idx < 0 || entry.map.set_idx >= cfg_.geometry.num_sets) {
        continue;
      }
      SetState& set = sets_[static_cast<std::size_t>(entry.map.set_idx)];
      const int existing = replacement_->FindWay(set, entry.map.tag);
      if (existing >= 0) {
        replacement_->OnPrefetchTouch(set, existing);
        LineMeta& line = set.lines[static_cast<std::size_t>(existing)];
        line.last_timestep = -1;
        line.output_spine_id = -1;
        continue;
      }
      VictimInfo victim = replacement_->PickVictim(set);
      if (victim.way < 0 || victim.way >= cfg_.geometry.ways) {
        continue;
      }
      if (victim.was_valid) {
        stats_.evictions_total += 1;
      }
      replacement_->Install(set, victim.way, entry.map.tag, tile_id);
      LineMeta& new_line = set.lines[static_cast<std::size_t>(victim.way)];
      new_line.output_spine_id = -1;
      new_line.last_timestep = -1;
      stats_.prefetch_inserts += 1;
      stats_.prefetch_bytes_loaded += static_cast<std::uint64_t>(cfg_.geometry.line_size_bytes);
    }
    prefetch_buffer_->Clear();
  }

  const CacheStats& Stats() const override {
    return stats_;
  }

private:
  CacheConfig cfg_;
  std::unique_ptr<IMapper> mapper_;
  std::unique_ptr<IReplacement> replacement_;
  std::unique_ptr<IPrefetch> prefetch_;
  std::unique_ptr<IWindow> window_;
  std::unique_ptr<PrefetchBuffer> prefetch_buffer_;

  std::vector<SetState> sets_;
  CacheStats stats_{};
  std::optional<AccessResult> last_access_;
  int last_tile_ = -1;
};

std::unique_ptr<ICache> BuildCache(const CacheConfig& cfg,
                                   CacheModules modules) {
  std::unique_ptr<ICache> cache = std::make_unique<CacheCore>(cfg, std::move(modules));
  if (auto decorator = GetCacheDecorator()) {
    cache = decorator(std::move(cache));
  }
  return cache;
}

} // namespace sf::cache
