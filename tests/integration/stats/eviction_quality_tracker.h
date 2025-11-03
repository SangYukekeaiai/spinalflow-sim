// All comments are in English.
#pragma once

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "cache/cache_hooks.h"
#include "cache/cache_iface.h"

namespace test::stats {

class EvictionQualityTracker {
public:
  void Reset() {
    pending_.clear();
    buckets_.clear();
    max_delta_ = 0;
    aggregated_.clear();
    aggregated_max_delta_ = 0;
  }

  void Record(const sf::cache::AccessRequest& request,
              const sf::cache::AccessResult& result) {
    if (result.set_idx < 0) {
      return;
    }
    Key key{result.set_idx, result.tag};
    auto it = pending_.find(key);
    if (it != pending_.end() && request.tile_id >= 0 && request.output_spine_id >= 0) {
      auto& records = it->second;
      for (auto rit = records.rbegin(); rit != records.rend(); ++rit) {
        if (rit->output_spine_id != request.output_spine_id) {
          continue;
        }
        if (rit->tile_id != request.tile_id) {
          continue;
        }
        const int reuse_ts = (request.timestep >= 0) ? request.timestep : rit->timestep;
        int delta = reuse_ts - rit->timestep;
        if (delta < 1) {
          delta = 1;
        }
        auto bkey = std::make_tuple(rit->output_spine_id, rit->tile_id, rit->timestep);
        auto& bucket = buckets_[bkey];
        bucket.bad_total += 1;
        bucket.histogram[delta] += 1;
        if (delta > max_delta_) {
          max_delta_ = delta;
        }
        auto agg_key = std::make_pair(rit->output_spine_id, rit->tile_id);
        auto& agg_bucket = aggregated_[agg_key];
        agg_bucket.bad_total += 1;
        agg_bucket.histogram[delta] += 1;
        if (delta > aggregated_max_delta_) {
          aggregated_max_delta_ = delta;
        }
        records.erase(std::next(rit).base());
        break;
      }
      if (records.empty()) {
        pending_.erase(it);
      }
    }

    if (result.evicted) {
      const int evict_spine = (result.evicted_output_spine_id >= 0)
                                  ? result.evicted_output_spine_id
                                  : request.output_spine_id;
      const int evict_tile = (result.evicted_tile_id >= 0)
                                 ? result.evicted_tile_id
                                 : request.tile_id;
      const int evict_ts = (request.timestep >= 0) ? request.timestep : result.evicted_last_timestep;
      if (evict_spine >= 0 && evict_tile >= 0 && evict_ts >= 0) {
        auto bkey = std::make_tuple(evict_spine, evict_tile, evict_ts);
        auto& bucket = buckets_[bkey];
        bucket.eviction_total += 1;
        pending_[{result.set_idx, result.evicted_tag}].push_back(
            EvictionRecord{evict_spine, evict_tile, evict_ts});
      }
    }
  }

  void WriteCsv(const std::filesystem::path& path) const {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream ofs(path, std::ios::out | std::ios::trunc);
    if (!ofs) {
      return;
    }
    ofs << "output spine id,tile id,time steps,eviction total,bad eviction total";
    for (int i = 1; i <= std::max(1, max_delta_); ++i) {
      ofs << ",bad eviction " << i << " times";
    }
    ofs << '\n';

    for (const auto& [key, bucket] : buckets_) {
      const auto& [spine, tile, timestep] = key;
      ofs << spine
          << ',' << tile
          << ',' << timestep
          << ',' << bucket.eviction_total
          << ',' << bucket.bad_total;
      for (int i = 1; i <= std::max(1, max_delta_); ++i) {
        auto hit = bucket.histogram.find(i);
        ofs << ',' << (hit != bucket.histogram.end() ? hit->second : 0);
      }
      ofs << '\n';
    }
  }

  void WriteAggregateCsv(const std::filesystem::path& path) const {
    if (aggregated_.empty()) {
      return;
    }
    std::filesystem::create_directories(path.parent_path());
    std::ofstream ofs(path, std::ios::out | std::ios::trunc);
    if (!ofs) {
      return;
    }
    ofs << "output spine id,tile id,bad eviction total";
    for (int i = 1; i <= std::max(1, aggregated_max_delta_); ++i) {
      ofs << ",bad eviction " << i << " times";
    }
    ofs << '\n';

    for (const auto& [key, bucket] : aggregated_) {
      ofs << key.first
          << ',' << key.second
          << ',' << bucket.bad_total;
      for (int i = 1; i <= std::max(1, aggregated_max_delta_); ++i) {
        auto hit = bucket.histogram.find(i);
        ofs << ',' << (hit != bucket.histogram.end() ? hit->second : 0);
      }
      ofs << '\n';
    }
  }

private:
  struct EvictionRecord {
    int output_spine_id = -1;
    int tile_id = -1;
    int timestep = -1;
  };

  struct Bucket {
    std::uint64_t eviction_total = 0;
    std::uint64_t bad_total = 0;
    std::map<int, std::uint64_t> histogram;
  };

  struct AggregateBucket {
    std::uint64_t bad_total = 0;
    std::map<int, std::uint64_t> histogram;
  };

  struct Key {
    int set_idx = -1;
    std::uint64_t tag = 0;

    bool operator==(const Key& other) const noexcept {
      return set_idx == other.set_idx && tag == other.tag;
    }
  };

  struct KeyHash {
    std::size_t operator()(const Key& key) const noexcept {
      return std::hash<int>()(key.set_idx) ^ (std::hash<std::uint64_t>()(key.tag) << 1);
    }
  };

  std::unordered_map<Key, std::vector<EvictionRecord>, KeyHash> pending_;
  std::map<std::tuple<int, int, int>, Bucket> buckets_;
  std::map<std::pair<int, int>, AggregateBucket> aggregated_;
  int max_delta_ = 0;
  int aggregated_max_delta_ = 0;
};

class EvictionTrackingCache final : public sf::cache::ICache {
public:
  EvictionTrackingCache(std::unique_ptr<sf::cache::ICache> inner,
                        EvictionQualityTracker* tracker)
      : inner_(std::move(inner)),
        tracker_(tracker) {}

  void Reset() override { inner_->Reset(); }
  void SetWindow(int cur_tile, int next_tile) override {
    inner_->SetWindow(cur_tile, next_tile);
  }
  void AdvanceWindow() override { inner_->AdvanceWindow(); }

  sf::cache::AccessResult OnDemandAccess(const sf::cache::AccessRequest& request) override {
    auto result = inner_->OnDemandAccess(request);
    if (tracker_) {
      tracker_->Record(request, result);
    }
    return result;
  }

  const sf::cache::CacheStats& Stats() const override {
    return inner_->Stats();
  }

private:
  std::unique_ptr<sf::cache::ICache> inner_;
  EvictionQualityTracker* tracker_ = nullptr;
};

class ScopedEvictionTracking {
public:
  explicit ScopedEvictionTracking(EvictionQualityTracker& tracker)
      : tracker_(tracker),
        previous_(sf::cache::GetCacheDecorator()) {
    sf::cache::RegisterCacheDecorator(
        [this](std::unique_ptr<sf::cache::ICache> cache) {
          if (previous_) {
            cache = previous_(std::move(cache));
          }
          return std::make_unique<EvictionTrackingCache>(std::move(cache), &tracker_);
        });
  }

  ~ScopedEvictionTracking() {
    sf::cache::RegisterCacheDecorator(previous_);
  }

  ScopedEvictionTracking(const ScopedEvictionTracking&) = delete;
  ScopedEvictionTracking& operator=(const ScopedEvictionTracking&) = delete;

private:
  EvictionQualityTracker& tracker_;
  sf::cache::CacheDecorator previous_;
};

} // namespace test::stats
