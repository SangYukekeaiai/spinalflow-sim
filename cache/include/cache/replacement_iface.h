// All comments are in English.
#pragma once

#include <cstdint>
#include <vector>

namespace sf::cache {

enum class Residency : std::uint8_t {
  Invalid = 0,
  Probation,
  Protected
};

struct LineMeta {
  bool valid = false;
  std::uint64_t tag = 0;
  int touches = 0;
  int tile_id = -1;
  Residency residency = Residency::Invalid;
};

struct SetState {
  std::vector<LineMeta> lines;
  std::vector<int> probation_order; // MRU at index 0
  std::vector<int> protected_order; // MRU at index 0
};

struct VictimInfo {
  int way = -1;
  bool was_valid = false;
  Residency residency = Residency::Invalid;
};

class IReplacement {
public:
  virtual ~IReplacement() = default;
  virtual void InitSet(SetState& set, int ways) = 0;
  virtual void ResetSet(SetState& set) = 0;
  virtual int FindWay(const SetState& set, std::uint64_t tag) const = 0;
  virtual VictimInfo PickVictim(SetState& set) = 0;
  virtual void Install(SetState& set, int way, std::uint64_t tag, int tile_id) = 0;
  virtual void OnHit(SetState& set, int way) = 0;
  virtual void OnPrefetchTouch(SetState& set, int way) = 0;
};

} // namespace sf::cache

