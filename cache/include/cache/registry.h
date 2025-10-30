// All comments are in English.
#pragma once

#include <memory>

#include "cache/cache_config.h"
#include "cache/mapper_iface.h"
#include "cache/replacement_iface.h"
#include "cache/prefetch_iface.h"
#include "cache/window_iface.h"

namespace sf::cache {

struct CacheModules {
  std::unique_ptr<IMapper> mapper;
  std::unique_ptr<IReplacement> replacement;
  std::unique_ptr<IPrefetch> prefetch;
  std::unique_ptr<IWindow> window;
};

CacheModules MakeDefaultModules(const CacheConfig& cfg);

std::unique_ptr<IReplacement> MakeTemporalAwareReplacement(const CacheConfig& cfg);

} // namespace sf::cache
