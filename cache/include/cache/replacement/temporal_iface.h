// All comments are in English.
#pragma once

#include "cache/replacement_iface.h"
#include "cache/mapper_iface.h"
#include "cache/cache_iface.h"

namespace sf::cache {

class ITemporalReplacement {
public:
  virtual ~ITemporalReplacement() = default;

  // Called whenever the active tile changes. total_tiles may be zero if unknown.
  virtual void OnTileStart(int new_tile, int prev_tile) = 0;

  // Returns true if the temporal buffer served the access as a hit.
  virtual bool TryTemporalHit(const AccessRequest& request,
                              const MapOutput& map,
                              AccessResult& result) = 0;

  // Handles a demand miss for the current tile. Returns true if the miss was
  // captured/promoted entirely inside the policy (no further action required).
  virtual bool HandleCurTileMiss(SetState& set,
                                 const AccessRequest& request,
                                 const MapOutput& map,
                                 AccessResult& result) = 0;
};

} // namespace sf::cache
