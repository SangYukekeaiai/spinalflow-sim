// All comments are in English.
#pragma once

#include <functional>
#include <memory>

#include "cache/cache_iface.h"

namespace sf::cache {

using CacheDecorator =
    std::function<std::unique_ptr<ICache>(std::unique_ptr<ICache>)>;

void RegisterCacheDecorator(CacheDecorator decorator);
void ClearCacheDecorator();
CacheDecorator GetCacheDecorator();

} // namespace sf::cache

