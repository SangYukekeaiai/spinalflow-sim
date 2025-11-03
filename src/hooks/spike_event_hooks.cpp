// All comments are in English.
#include "hooks/spike_event_hooks.h"

namespace sf::hooks {

namespace {

SpikeEventCallback& CallbackStorage() {
  static SpikeEventCallback cb;
  return cb;
}

} // namespace

void RegisterSpikeEventCallback(SpikeEventCallback cb) {
  CallbackStorage() = std::move(cb);
}

void ClearSpikeEventCallback() {
  CallbackStorage() = nullptr;
}

SpikeEventCallback GetSpikeEventCallback() {
  return CallbackStorage();
}

} // namespace sf::hooks

