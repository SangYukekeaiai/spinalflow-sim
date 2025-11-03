// All comments are in English.
#pragma once

#include <cstdint>
#include <functional>

namespace sf::hooks {

using TileInputCallback =
    std::function<void(int output_spine_id, int tile_id, std::uint64_t neuron_id)>;

void RegisterTileInputCallback(TileInputCallback cb);
void ClearTileInputCallback();
TileInputCallback GetTileInputCallback();

} // namespace sf::hooks

