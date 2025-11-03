// All comments are in English.
#include "cache/belady.h"

#include <cassert>
#include <limits>
#include <stdexcept>
#include <unordered_map>

#include "cache/cache_config.h"
#include "cache/cache_iface.h"
#include "cache/mapper_iface.h"

namespace sf::cache {

std::unique_ptr<IMapper> MakeXorFoldMapper(const CacheConfig& cfg);
std::unique_ptr<IMapper> MakeDirectModMapper(const CacheConfig& cfg);

namespace {

std::shared_ptr<const BeladyPlan>& PlanStorage() {
  static std::shared_ptr<const BeladyPlan> plan;
  return plan;
}

} // namespace

std::shared_ptr<const BeladyPlan> BuildBeladyPlan(const CacheConfig& cfg,
                                                  const std::vector<AccessRequest>& trace) {
  CacheConfig materialized = cfg;
  materialized.Validate();

  if (materialized.geometry.num_sets <= 0) {
    throw std::invalid_argument("BuildBeladyPlan: num_sets must be positive.");
  }

  std::unique_ptr<IMapper> mapper = MakeXorFoldMapper(materialized);
  if (!mapper) {
    mapper = MakeDirectModMapper(materialized);
  }
  if (!mapper) {
    throw std::runtime_error("BuildBeladyPlan: mapper construction failed.");
  }

  BeladyPlan plan;
  plan.num_sets = materialized.geometry.num_sets;
  plan.accesses.reserve(trace.size());

  for (const auto& request : trace) {
    MapOutput map = mapper->Map({request.tile_id, request.cin, request.kh, request.kw});
    if (map.set_idx < 0 || map.set_idx >= materialized.geometry.num_sets) {
      throw std::runtime_error("BuildBeladyPlan: mapper produced out-of-range set index.");
    }
    BeladyAccessInfo info;
    info.set_idx = map.set_idx;
    info.tag = map.tag;
    plan.accesses.push_back(info);
  }

  if (plan.accesses.empty()) {
    return std::make_shared<BeladyPlan>(std::move(plan));
  }

  std::vector<std::unordered_map<std::uint64_t, std::size_t>> next_use(
      static_cast<std::size_t>(plan.num_sets));

  for (std::size_t idx = plan.accesses.size(); idx-- > 0;) {
    auto& entry = plan.accesses[idx];
    auto& per_set = next_use[static_cast<std::size_t>(entry.set_idx)];
    auto it = per_set.find(entry.tag);
    if (it == per_set.end()) {
      entry.next_use_index = kBeladyNoFutureUse;
    } else {
      entry.next_use_index = it->second;
    }
    per_set[entry.tag] = idx;
  }

  return std::make_shared<const BeladyPlan>(std::move(plan));
}

void RegisterBeladyPlan(std::shared_ptr<const BeladyPlan> plan) {
  PlanStorage() = std::move(plan);
}

std::shared_ptr<const BeladyPlan> GetBeladyPlan() {
  return PlanStorage();
}

void ClearBeladyPlan() {
  PlanStorage().reset();
}

} // namespace sf::cache
