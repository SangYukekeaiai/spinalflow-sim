// All comments are in English.
#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <unordered_map>
#include <vector>

namespace test::stats {

class ReuseInTileTracker {
public:
  void BeginLayer(int layer_id, int tiles_per_spine) {
    layer_id_ = layer_id;
    tiles_per_spine_ = tiles_per_spine;
    active_ = true;
    max_reuse_ = 0;
    records_.clear();
  }

  void Record(int spine_id, int tile_id, std::uint64_t address) {
    if (!active_ || layer_id_ < 0) return;
    auto& tile_maps = records_[spine_id];
    if (static_cast<int>(tile_maps.size()) < tiles_per_spine_) {
      tile_maps.resize(static_cast<std::size_t>(tiles_per_spine_));
    }
    auto& counts = tile_maps[static_cast<std::size_t>(tile_id)];
    std::uint32_t& entry = counts[address];
    entry += 1;
    if (entry > max_reuse_) {
      max_reuse_ = entry;
    }
  }

  void WriteCsv(const std::filesystem::path& path) {
    if (layer_id_ < 0 || !active_) return;

    std::filesystem::create_directories(path.parent_path());
    std::ofstream ofs(path, std::ios::out | std::ios::trunc);
    if (!ofs) return;

    const std::uint32_t max_reuse = max_reuse_ > 0 ? max_reuse_ : 0;
    ofs << "output_spine_id,tile_id";
    for (std::uint32_t r = 1; r <= max_reuse; ++r) {
      ofs << ',' << r;
    }
    ofs << '\n';

    for (auto& [spine_id, tile_maps] : records_) {
      if (tile_maps.empty()) {
        tile_maps.resize(static_cast<std::size_t>(tiles_per_spine_));
      }
      for (int tile = 0; tile < tiles_per_spine_; ++tile) {
        ofs << spine_id << ',' << tile;
        const auto& counts = tile_maps[static_cast<std::size_t>(tile)];
        std::vector<std::uint64_t> buckets((max_reuse ? max_reuse + 1 : 1), 0);
        for (const auto& kv : counts) {
          std::uint32_t reuse = kv.second;
          if (reuse >= buckets.size()) {
            buckets.resize(reuse + 1, 0);
          }
          buckets[reuse] += 1;
        }
        for (std::uint32_t r = 1; r <= max_reuse; ++r) {
          std::uint64_t value = (r < buckets.size()) ? buckets[r] : 0;
          ofs << ',' << value;
        }
        ofs << '\n';
      }
    }

    active_ = false;
    layer_id_ = -1;
    max_reuse_ = 0;
    records_.clear();
  }

private:
  using AddressCounts = std::unordered_map<std::uint64_t, std::uint32_t>;
  using TileRecords = std::vector<AddressCounts>;

  int layer_id_ = -1;
  int tiles_per_spine_ = 0;
  bool active_ = false;
  std::uint32_t max_reuse_ = 0;
  std::unordered_map<int, TileRecords> records_;
};

} // namespace test::stats

