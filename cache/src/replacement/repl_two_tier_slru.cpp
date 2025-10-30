// All comments are in English.
#include "cache/replacement_iface.h"

#include <algorithm>
#include <memory>

namespace sf::cache {

namespace {
constexpr int kTouchSaturate = 31;

void EraseWay(std::vector<int>& order, int way) {
  order.erase(std::remove(order.begin(), order.end(), way), order.end());
}

} // namespace

class TwoTierSlruReplacement final : public IReplacement {
public:
  void InitSet(SetState& set, int ways) override {
    set.lines.resize(static_cast<std::size_t>(ways));
    set.probation_order.clear();
    set.protected_order.clear();
  }

  void ResetSet(SetState& set) override {
    for (auto& line : set.lines) {
      line = {};
    }
    set.probation_order.clear();
    set.protected_order.clear();
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
    for (int way = 0; way < static_cast<int>(set.lines.size()); ++way) {
      LineMeta& line = set.lines[static_cast<std::size_t>(way)];
      if (!line.valid) {
        EraseWay(set.probation_order, way);
        EraseWay(set.protected_order, way);
        return VictimInfo{way, false, Residency::Invalid};
      }
    }

    if (!set.probation_order.empty()) {
      const int victim = set.probation_order.back();
      set.probation_order.pop_back();
      LineMeta& line = set.lines[static_cast<std::size_t>(victim)];
      return VictimInfo{victim, line.valid, line.residency};
    }

    if (!set.protected_order.empty()) {
      const int victim = set.protected_order.back();
      set.protected_order.pop_back();
      LineMeta& line = set.lines[static_cast<std::size_t>(victim)];
      return VictimInfo{victim, line.valid, line.residency};
    }

    return VictimInfo{0, set.lines[0].valid, set.lines[0].residency};
  }

  void Install(SetState& set, int way, std::uint64_t tag, int tile_id) override {
    LineMeta& line = set.lines[static_cast<std::size_t>(way)];
    EraseWay(set.probation_order, way);
    EraseWay(set.protected_order, way);
    line.valid = true;
    line.tag = tag;
    line.tile_id = tile_id;
    line.touches = 1;
    line.residency = Residency::Probation;
    set.probation_order.insert(set.probation_order.begin(), way);
  }

  void OnHit(SetState& set, int way) override {
    if (way < 0 || way >= static_cast<int>(set.lines.size())) return;
    LineMeta& line = set.lines[static_cast<std::size_t>(way)];
    if (line.touches < kTouchSaturate) {
      line.touches += 1;
    }
    if (line.residency == Residency::Probation) {
      if (line.touches >= 2) {
        PromoteToProtected(set, way, line);
      } else {
        RefreshProbation(set, way);
      }
    } else if (line.residency == Residency::Protected) {
      RefreshProtected(set, way);
    }
  }

  void OnPrefetchTouch(SetState& set, int way) override {
    if (way < 0 || way >= static_cast<int>(set.lines.size())) return;
    LineMeta& line = set.lines[static_cast<std::size_t>(way)];
    if (line.residency == Residency::Probation) {
      RefreshProbation(set, way);
    } else if (line.residency == Residency::Protected) {
      RefreshProtected(set, way);
    }
  }

private:
  void RefreshProbation(SetState& set, int way) {
    EraseWay(set.probation_order, way);
    set.probation_order.insert(set.probation_order.begin(), way);
  }

  void RefreshProtected(SetState& set, int way) {
    EraseWay(set.protected_order, way);
    set.protected_order.insert(set.protected_order.begin(), way);
  }

  void PromoteToProtected(SetState& set, int way, LineMeta& line) {
    EraseWay(set.probation_order, way);
    line.residency = Residency::Protected;
    if (line.touches < 2) {
      line.touches = 2;
    }
    set.protected_order.insert(set.protected_order.begin(), way);
  }
};

std::unique_ptr<IReplacement> MakeTwoTierSlruReplacement() {
  return std::make_unique<TwoTierSlruReplacement>();
}

} // namespace sf::cache
