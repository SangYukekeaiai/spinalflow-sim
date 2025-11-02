// All comments are in English.
#include "cache/replacement_iface.h"

#include <algorithm>
#include <cassert>
#include <memory>
#include <numeric>
#include <unordered_map>

namespace sf::cache {

namespace {

std::vector<int>& LruOrderFor(std::unordered_map<const SetState*, std::vector<int>>& table,
                              SetState& set,
                              int ways) {
  auto& order = table[&set];
  if (static_cast<int>(order.size()) != ways) {
    order.resize(static_cast<std::size_t>(ways));
    std::iota(order.begin(), order.end(), 0);
  }
  return order;
}

} // namespace

class LruReplacement final : public IReplacement {
public:
  void InitSet(SetState& set, int ways) override {
    set.lines.resize(static_cast<std::size_t>(ways));
    auto& order = LruOrderFor(order_, set, ways);
    std::iota(order.begin(), order.end(), 0);
    assert(order.size() == static_cast<std::size_t>(ways));
  }

  void ResetSet(SetState& set) override {
    for (auto& line : set.lines) {
      line = {};
    }
    auto it = order_.find(&set);
    if (it != order_.end()) {
      auto& order = it->second;
      order.resize(set.lines.size());
      std::iota(order.begin(), order.end(), 0);
      assert(order.size() == set.lines.size());
    }
  }

  int FindWay(const SetState& set, std::uint64_t tag) const override {
    for (int way = 0; way < static_cast<int>(set.lines.size()); ++way) {
      const LineMeta& line = set.lines[static_cast<std::size_t>(way)];
      if (line.valid && line.tag == tag) {
        return way;
      }
    }
    return -1;
  }

  VictimInfo PickVictim(SetState& set) override {
    const int ways = static_cast<int>(set.lines.size());
    auto& order = LruOrderFor(order_, set, ways);

    for (int way = 0; way < ways; ++way) {
      LineMeta& line = set.lines[static_cast<std::size_t>(way)];
      if (!line.valid) {
        Touch(order, way);
        return VictimInfo{way, false};
      }
    }

    if (order.empty()) {
      return VictimInfo{};
    }

    const int victim = order.back();
    order.pop_back();
    order.insert(order.begin(), victim);
    assert(order.size() == static_cast<std::size_t>(ways));

    LineMeta& line = set.lines[static_cast<std::size_t>(victim)];
    return VictimInfo{victim, line.valid};
  }

  void Install(SetState& set, int way, std::uint64_t tag, int tile_id) override {
    LineMeta& line = set.lines[static_cast<std::size_t>(way)];
    line.valid = true;
    line.tag = tag;
    line.tile_id = tile_id;

    auto& order = order_[&set];
    Touch(order, way);
  }

  void OnHit(SetState& set, int way) override {
    auto& order = order_[&set];
    Touch(order, way);
  }

  void OnPrefetchTouch(SetState& set, int way) override {
    auto& order = order_[&set];
    Touch(order, way);
    assert(order.size() == set.lines.size());
  }

private:
  void Touch(std::vector<int>& order, int way) {
    order.erase(std::remove(order.begin(), order.end(), way), order.end());
    order.insert(order.begin(), way);
    assert(!order.empty());
  }

  std::unordered_map<const SetState*, std::vector<int>> order_;
};

std::unique_ptr<IReplacement> MakeLruReplacement() {
  return std::make_unique<LruReplacement>();
}

} // namespace sf::cache
