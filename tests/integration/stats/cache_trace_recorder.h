// All comments are in English.
#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "cache/cache_hooks.h"
#include "cache/cache_iface.h"

namespace test::stats {

class TraceRecordingCache final : public sf::cache::ICache {
public:
  TraceRecordingCache(std::unique_ptr<sf::cache::ICache> inner,
                      std::vector<sf::cache::AccessRequest>* trace)
      : inner_(std::move(inner)),
        trace_(trace) {}

  void Reset() override { inner_->Reset(); }
  void SetWindow(int cur_tile, int next_tile) override {
    inner_->SetWindow(cur_tile, next_tile);
  }
  void AdvanceWindow() override { inner_->AdvanceWindow(); }

  sf::cache::AccessResult OnDemandAccess(const sf::cache::AccessRequest& request) override {
    if (trace_) {
      trace_->push_back(request);
    }
    return inner_->OnDemandAccess(request);
  }

  const sf::cache::CacheStats& Stats() const override {
    return inner_->Stats();
  }

private:
  std::unique_ptr<sf::cache::ICache> inner_;
  std::vector<sf::cache::AccessRequest>* trace_ = nullptr;
};

class ScopedTraceRecorder {
public:
  explicit ScopedTraceRecorder(std::vector<sf::cache::AccessRequest>& trace)
      : trace_(trace),
        previous_(sf::cache::GetCacheDecorator()) {
    trace_.clear();
    sf::cache::RegisterCacheDecorator(
        [this](std::unique_ptr<sf::cache::ICache> cache) {
          if (previous_) {
            cache = previous_(std::move(cache));
          }
          return std::make_unique<TraceRecordingCache>(std::move(cache), &trace_);
        });
  }

  ~ScopedTraceRecorder() {
    sf::cache::RegisterCacheDecorator(previous_);
  }

  ScopedTraceRecorder(const ScopedTraceRecorder&) = delete;
  ScopedTraceRecorder& operator=(const ScopedTraceRecorder&) = delete;

private:
  std::vector<sf::cache::AccessRequest>& trace_;
  sf::cache::CacheDecorator previous_;
};

} // namespace test::stats

