// All comments are in English.
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "cache/cache_config.h"
#include "cache/cache_iface.h"
#include "cache/replacement_iface.h"

namespace sf::cache {

struct BeladyAccessInfo {
  int set_idx = -1;
  std::uint64_t tag = 0;
  std::size_t next_use_index = std::numeric_limits<std::size_t>::max();
};

struct BeladyPlan {
  int num_sets = 0;
  std::vector<BeladyAccessInfo> accesses;
};

std::shared_ptr<const BeladyPlan> BuildBeladyPlan(const CacheConfig& cfg,
                                                  const std::vector<AccessRequest>& trace);

void RegisterBeladyPlan(std::shared_ptr<const BeladyPlan> plan);
std::shared_ptr<const BeladyPlan> GetBeladyPlan();
void ClearBeladyPlan();

std::unique_ptr<IReplacement> MakeBeladyReplacement(const CacheConfig& cfg);

inline constexpr std::size_t kBeladyNoFutureUse = std::numeric_limits<std::size_t>::max();

} // namespace sf::cache

