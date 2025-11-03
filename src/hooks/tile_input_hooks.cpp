// All comments are in English.
#include "hooks/tile_input_hooks.h"

namespace sf::hooks {

namespace {

TileInputCallback& CallbackStorage() {
  static TileInputCallback cb;
  return cb;
}

} // namespace

void RegisterTileInputCallback(TileInputCallback cb) {
  CallbackStorage() = std::move(cb);
}

void ClearTileInputCallback() {
  CallbackStorage() = nullptr;
}

TileInputCallback GetTileInputCallback() {
  return CallbackStorage();
}

} // namespace sf::hooks

