// All comments are in English.
#include "cache/prefetch_iface.h"

#include <memory>

namespace sf::cache {

class ZeroLatencyNextTilePrefetch final : public IPrefetch {
public:
  explicit ZeroLatencyNextTilePrefetch(const CacheConfig& cfg)
      : cfg_(cfg),
        S_tile_(cfg.Cin * cfg.KH * cfg.KW),
        N_color_(cfg.geometry.num_sets / 2) {}

  PrefetchPlan Plan(const PrefetchInput& input) const override {
    PrefetchPlan plan;
    if (input.cur_tile < 0 || input.next_tile < 0) {
      return plan;
    }
    if (input.request.tile_id != input.cur_tile) {
      return plan;
    }
    if (input.next_tile != input.cur_tile + 1) {
      return plan;
    }
    if (input.map.L < 0 || input.map.L >= S_tile_) {
      return plan;
    }
    plan.do_prefetch = true;
    plan.tile_id = input.next_tile;
    plan.target_map.L = input.map.L;
    plan.target_map.tag =
        static_cast<std::uint64_t>(static_cast<long long>(plan.tile_id) * S_tile_) +
        static_cast<std::uint64_t>(plan.target_map.L);
    const int idx_in_color = (static_cast<long long>(cfg_.A1) * plan.target_map.L) % N_color_;
    const int parity = plan.tile_id & 1;
    plan.target_map.set_idx = (idx_in_color << 1) | parity;
    return plan;
  }

private:
  CacheConfig cfg_;
  int S_tile_ = 0;
  int N_color_ = 0;
};

std::unique_ptr<IPrefetch> MakeZeroLatencyNextTilePrefetch(const CacheConfig& cfg) {
  return std::make_unique<ZeroLatencyNextTilePrefetch>(cfg);
}

} // namespace sf::cache

