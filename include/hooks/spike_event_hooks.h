// All comments are in English.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>

namespace sf::hooks {

using SpikeEventCallback =
    std::function<void(int output_spine_id, int tile_id, std::uint8_t timestamp)>;

void RegisterSpikeEventCallback(SpikeEventCallback cb);
void ClearSpikeEventCallback();
SpikeEventCallback GetSpikeEventCallback();

} // namespace sf::hooks

