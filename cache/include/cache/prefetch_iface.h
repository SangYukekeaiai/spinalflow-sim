// All comments are in English.
#pragma once

#include <optional>

#include "cache/cache_config.h"
#include "cache/cache_iface.h"
#include "cache/mapper_iface.h"

namespace sf::cache {

struct PrefetchInput {
  int cur_tile = -1;
  int next_tile = -1;
  MapOutput map{};
  AccessRequest request{};
};

struct PrefetchPlan {
  bool do_prefetch = false;
  MapOutput target_map{};
  int tile_id = -1;
};

class IPrefetch {
public:
  virtual ~IPrefetch() = default;
  virtual PrefetchPlan Plan(const PrefetchInput& input) const = 0;
};

} // namespace sf::cache

