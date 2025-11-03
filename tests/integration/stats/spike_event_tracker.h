// All comments are in English.
#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <unordered_map>
#include <vector>

namespace test::stats {

class SpikeEventTracker {
public:
  void BeginLayer(int layer_id, int tiles_per_spine) {
    layer_id_ = layer_id;
    tiles_per_spine_ = tiles_per_spine;
    active_ = true;
    records_.clear();
    max_ts_ = -1;
  }

  void Record(int spine_id, int tile_id, std::uint8_t ts) {
    if (!active_ || layer_id_ < 0) return;
    auto& tile_map = records_[spine_id];
    auto& timeline = tile_map[tile_id];
    if (ts >= timeline.size()) {
      timeline.resize(static_cast<std::size_t>(ts) + 1, 0);
    }
    timeline[static_cast<std::size_t>(ts)] += 1;
    if (ts > max_ts_) {
      max_ts_ = ts;
    }
  }

  void WriteCsv(const std::filesystem::path& path) {
    if (layer_id_ < 0) return;
    std::filesystem::create_directories(path.parent_path());
    std::ofstream ofs(path, std::ios::out | std::ios::trunc);
    if (!ofs) return;

    const int columns = (max_ts_ >= 0) ? (static_cast<int>(max_ts_) + 1) : 0;
    ofs << "output_spine_id,tile_id";
    for (int t = 0; t < columns; ++t) {
      ofs << ",t" << t;
    }
    ofs << '\n';

    for (const auto& [spine_id, tiles] : records_) {
      for (int tile = 0; tile < tiles_per_spine_; ++tile) {
        ofs << spine_id << ',' << tile;
        auto tile_it = tiles.find(tile);
        for (int t = 0; t < columns; ++t) {
          std::uint64_t value = 0;
          if (tile_it != tiles.end()) {
            const auto& timeline = tile_it->second;
            if (static_cast<std::size_t>(t) < timeline.size()) {
              value = timeline[static_cast<std::size_t>(t)];
            }
          }
          ofs << ',' << value;
        }
        ofs << '\n';
      }
    }

    active_ = false;
    layer_id_ = -1;
    max_ts_ = -1;
    records_.clear();
  }

private:
  using Timeline = std::vector<std::uint64_t>;
  using TileMap = std::map<int, Timeline>;

  int layer_id_ = -1;
  int tiles_per_spine_ = 0;
  bool active_ = false;
  int max_ts_ = -1;
  std::map<int, TileMap> records_;
};

} // namespace test::stats

