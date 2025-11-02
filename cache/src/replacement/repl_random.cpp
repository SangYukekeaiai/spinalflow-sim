// All comments are in English.
#include "cache/replacement_iface.h"

#include <cstdlib>
#include <memory>

namespace sf::cache {

class RandomReplacement final : public IReplacement {
public:
  void InitSet(SetState& set, int ways) override {
    set.lines.resize(static_cast<std::size_t>(ways));
  }

  void ResetSet(SetState& set) override {
    for (auto& line : set.lines) {
      line = {};
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
    for (int way = 0; way < static_cast<int>(set.lines.size()); ++way) {
      LineMeta& line = set.lines[static_cast<std::size_t>(way)];
      if (!line.valid) {
        return VictimInfo{way, false};
      }
    }
    const int victim = std::rand() % static_cast<int>(set.lines.size());
    LineMeta& line = set.lines[static_cast<std::size_t>(victim)];
    return VictimInfo{victim, line.valid};
  }

  void Install(SetState& set, int way, std::uint64_t tag, int tile_id) override {
    LineMeta& line = set.lines[static_cast<std::size_t>(way)];
    line.valid = true;
    line.tag = tag;
    line.tile_id = tile_id;
  }

  void OnHit(SetState&, int) override {}
  void OnPrefetchTouch(SetState&, int) override {}
};

std::unique_ptr<IReplacement> MakeRandomReplacement() {
  return std::make_unique<RandomReplacement>();
}

} // namespace sf::cache
