// All comments are in English.
#pragma once

#include <memory>

#include "cache/cache_config.h"
#include "cache/mapper_iface.h"
#include "cache/replacement_iface.h"
#include "cache/prefetch_iface.h"
#include "cache/window_iface.h"
#include "cache/prefetch_buffer.h"

namespace sf::cache {

struct CacheModules {
  std::unique_ptr<IMapper> mapper;
  std::unique_ptr<IReplacement> replacement;
  std::unique_ptr<IPrefetch> prefetch;
  std::unique_ptr<IWindow> window;
  std::unique_ptr<PrefetchBuffer> prefetch_buffer;
};

CacheModules MakeDefaultModules(const CacheConfig& cfg);

std::unique_ptr<IPrefetch> MakeFirstTouchPrefetch(const CacheConfig& cfg);

} // namespace sf::cache
