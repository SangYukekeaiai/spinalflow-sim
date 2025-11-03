// All comments are in English.
#pragma once

#include "hooks/spike_event_hooks.h"
#include "spike_event_tracker.h"

namespace test::stats {

inline void EnableSpikeEventTracking(SpikeEventTracker& tracker) {
  sf::hooks::RegisterSpikeEventCallback(
      [&tracker](int spine_id, int tile_id, std::uint8_t ts) {
        tracker.Record(spine_id, tile_id, ts);
      });
}

inline void DisableSpikeEventTracking() {
  sf::hooks::ClearSpikeEventCallback();
}

} // namespace test::stats
