// All comments are in English.
#pragma once

#include <string>
#include <vector>
#include <unordered_map>

#include "core/core.hpp"         // CoreCycleStats, CoreSramStats
#include "runner/simulation.hpp" // LayerKind

namespace sf {

// Lightweight record to aggregate per-layer stats for CSV emission.
struct LayerStageRecord {
  int layer_id = 0;
  std::string layer_name;
  LayerKind kind = LayerKind::kConv;
  CoreCycleStats cycles{};
  CoreSramStats sram_stats{};
};

} // namespace sf
