// All comments are in English.
#pragma once

#include "hooks/tile_input_hooks.h"
#include "reuse_in_tile_tracker.h"

namespace test::stats {

inline void EnableTileReuseTracking(ReuseInTileTracker& tracker) {
  sf::hooks::RegisterTileInputCallback(
      [&tracker](int spine_id, int tile_id, std::uint64_t address) {
        tracker.Record(spine_id, tile_id, address);
      });
}

inline void DisableTileReuseTracking() {
  sf::hooks::ClearTileInputCallback();
}

} // namespace test::stats
