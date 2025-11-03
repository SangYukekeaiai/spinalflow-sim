// All comments are in English.
#pragma once

#include <memory>

#include "cache/cache_hooks.h"
#include "cache/cache_iface.h"
#include "reuse_distance_tracker.h"

namespace test::stats {

class TrackingCache final : public sf::cache::ICache {
public:
  TrackingCache(std::unique_ptr<sf::cache::ICache> inner,
                ReuseDistanceTracker* tracker)
      : inner_(std::move(inner)),
        tracker_(tracker) {}

  void Reset() override { inner_->Reset(); }
  void SetWindow(int cur_tile, int next_tile) override {
    inner_->SetWindow(cur_tile, next_tile);
  }
  void AdvanceWindow() override { inner_->AdvanceWindow(); }

  sf::cache::AccessResult OnDemandAccess(const sf::cache::AccessRequest& request) override {
    if (tracker_) {
      tracker_->Record(request);
    }
    return inner_->OnDemandAccess(request);
  }

  const sf::cache::CacheStats& Stats() const override {
    return inner_->Stats();
  }

private:
  std::unique_ptr<sf::cache::ICache> inner_;
  ReuseDistanceTracker* tracker_ = nullptr;
};

inline void EnableReuseTracking(ReuseDistanceTracker& tracker) {
  sf::cache::RegisterCacheDecorator(
      [&tracker](std::unique_ptr<sf::cache::ICache> cache) {
        return std::make_unique<TrackingCache>(std::move(cache), &tracker);
      });
}

inline void DisableReuseTracking() {
  sf::cache::ClearCacheDecorator();
}

} // namespace test::stats
