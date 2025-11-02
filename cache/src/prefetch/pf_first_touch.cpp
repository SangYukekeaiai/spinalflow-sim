// All comments are in English.
#include "cache/prefetch_iface.h"

#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "cache/cache_config.h"

namespace sf::cache {

namespace {

using LineSet = std::unordered_set<int>;

class FirstTouchPrefetch final : public IPrefetch {
public:
  explicit FirstTouchPrefetch(const CacheConfig& cfg) {
    const long long tile_lines =
        static_cast<long long>(cfg.Cin) *
        static_cast<long long>(cfg.KH) *
        static_cast<long long>(cfg.KW);
    if (tile_lines <= 0) {
      throw std::invalid_argument("FirstTouchPrefetch: invalid tile geometry.");
    }
  }

  PrefetchPlan Plan(const PrefetchInput& input) const override {
    PrefetchPlan plan;
    const int tile = input.cur_tile;
    if (tile < 0) {
      return plan;
    }
    if (input.request.tile_id != tile) {
      return plan;
    }
    if (input.map.L < 0) {
      return plan;
    }

    if (first_tile_ < 0) {
      first_tile_ = tile;
    }

    if (tile != last_tile_) {
      touched_[tile].clear();
      last_tile_ = tile;
    }

    auto& touched_lines = touched_[tile];
    const bool first_touch = touched_lines.insert(input.map.L).second;
    if (!first_touch) {
      return plan;
    }

    int target_tile = input.next_tile;
    if (target_tile < 0) {
      target_tile = (first_tile_ >= 0) ? first_tile_ : 0;
    }

    if (target_tile < 0) {
      return plan;
    }

    plan.do_prefetch = true;
    plan.request = input.request;
    plan.request.tile_id = target_tile;
    return plan;
  }

private:
  mutable std::unordered_map<int, LineSet> touched_;
  mutable int last_tile_ = -1;
  mutable int first_tile_ = -1;
};

} // namespace

std::unique_ptr<IPrefetch> MakeFirstTouchPrefetch(const CacheConfig& cfg) {
  return std::make_unique<FirstTouchPrefetch>(cfg);
}

} // namespace sf::cache
