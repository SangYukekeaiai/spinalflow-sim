// All comments are in English.
#pragma once

#include <vector>

#include "cache/mapper_iface.h"

namespace sf::cache {

struct PrefetchBufferEntry {
  int tile_id = -1;
  MapOutput map{};
};

class PrefetchBuffer {
public:
  explicit PrefetchBuffer(int capacity_lines = 256);

  void Clear();
  bool Store(int tile_id, const MapOutput& map);
  std::vector<PrefetchBufferEntry> TakeForTile(int tile_id);
  int capacity() const { return capacity_; }

private:
  int capacity_ = 0;
  std::vector<PrefetchBufferEntry> entries_;
};

} // namespace sf::cache

