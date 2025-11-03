// All comments are in English.
#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <unordered_map>

#include "cache/cache_iface.h"

namespace test::stats {

class ReuseDistanceTracker {
public:
  void BeginLayer(int layer_id) {
    layer_id_ = layer_id;
    access_index_ = 0;
    last_seen_.clear();
    histogram_.clear();
  }

  void Record(const sf::cache::AccessRequest& request) {
    const std::uint64_t key = ComposeKey(request);
    auto it = last_seen_.find(key);
    if (it != last_seen_.end()) {
      const std::size_t distance = access_index_ - it->second;
      histogram_[distance] += 1;
    }
    last_seen_[key] = access_index_;
    access_index_ += 1;
  }

  void WriteCsv(const std::filesystem::path& path) const {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream ofs(path, std::ios::out | std::ios::trunc);
    ofs << "reuse_distance,count,share\n";
    const double total = static_cast<double>(TotalCount());
    if (total <= 0.0) {
      return;
    }
    for (const auto& [distance, count] : histogram_) {
      const double share = static_cast<double>(count) / total;
      ofs << distance << ',' << count << ',' << share << '\n';
    }
  }

  std::size_t TotalCount() const {
    std::size_t sum = 0;
    for (const auto& entry : histogram_) {
      sum += entry.second;
    }
    return sum;
  }

  int layer() const { return layer_id_; }

private:
  static std::uint64_t ComposeKey(const sf::cache::AccessRequest& req) {
    auto combine = [](std::uint64_t seed, int value) {
      const std::uint64_t v = static_cast<std::uint64_t>(static_cast<std::uint32_t>(value));
      seed ^= v + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
      return seed;
    };
    std::uint64_t key = 0;
    key = combine(key, req.tile_id);
    key = combine(key, req.cin);
    key = combine(key, req.kh);
    key = combine(key, req.kw);
    return key;
  }

  int layer_id_ = -1;
  std::size_t access_index_ = 0;
  std::unordered_map<std::uint64_t, std::size_t> last_seen_;
  std::map<std::size_t, std::uint64_t> histogram_;
};

} // namespace test::stats
