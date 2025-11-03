// All comments are in English.
#include "cache/prefetch_buffer.h"

#include <algorithm>

namespace sf::cache {

PrefetchBuffer::PrefetchBuffer(int capacity_lines)
    : capacity_(capacity_lines) {}

void PrefetchBuffer::Clear() {
  entries_.clear();
}

bool PrefetchBuffer::Store(int tile_id, const MapOutput& map) {
  if (capacity_ <= 0) return false;
  if (static_cast<int>(entries_.size()) >= capacity_) {
    return false;
  }
  // Avoid duplicates for the same tile/tag.
  const auto dup = std::find_if(entries_.begin(), entries_.end(),
                                [&](const PrefetchBufferEntry& e) {
                                  return e.tile_id == tile_id && e.map.tag == map.tag &&
                                         e.map.set_idx == map.set_idx;
                                });
  if (dup != entries_.end()) {
    return true;
  }
  entries_.push_back(PrefetchBufferEntry{tile_id, map});
  return true;
}

std::vector<PrefetchBufferEntry> PrefetchBuffer::TakeForTile(int tile_id) {
  std::vector<PrefetchBufferEntry> result;
  auto it = entries_.begin();
  while (it != entries_.end()) {
    if (it->tile_id == tile_id) {
      result.push_back(*it);
      it = entries_.erase(it);
    } else {
      ++it;
    }
  }
  return result;
}

} // namespace sf::cache

