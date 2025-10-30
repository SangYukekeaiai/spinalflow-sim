// All comments are in English.
#include "cache/replacement_iface.h"

#include <algorithm>
#include <memory>

namespace sf::cache {

class LruReplacement final : public IReplacement {
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
        TouchLru(set, way);
        return VictimInfo{way, false, Residency::Invalid};
      }
    }

    if (!set.probation_order.empty()) {
      const int victim = set.probation_order.back();
      set.probation_order.pop_back();
      LineMeta& line = set.lines[static_cast<std::size_t>(victim)];
      return VictimInfo{victim, line.valid, line.residency};
    }

    return VictimInfo{0, set.lines[0].valid, set.lines[0].residency};
  }

  void Install(SetState& set, int way, std::uint64_t tag, int tile_id) override {
    LineMeta& line = set.lines[static_cast<std::size_t>(way)];
    TouchLru(set, way);
    line.valid = true;
    line.tag = tag;
    line.tile_id = tile_id;
    line.touches = 1;
    line.residency = Residency::Probation;
  }

  void OnHit(SetState& set, int way) override {
    TouchLru(set, way);
  }

  void OnPrefetchTouch(SetState& set, int way) override {
    TouchLru(set, way);
  }

private:
  void TouchLru(SetState& set, int way) {
    set.probation_order.erase(std::remove(set.probation_order.begin(),
                                          set.probation_order.end(),
                                          way),
                              set.probation_order.end());
    set.probation_order.insert(set.probation_order.begin(), way);
  }
};

std::unique_ptr<IReplacement> MakeLruReplacement() {
  return std::make_unique<LruReplacement>();
}

} // namespace sf::cache

