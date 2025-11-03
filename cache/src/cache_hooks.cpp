// All comments are in English.
#include "cache/cache_hooks.h"

namespace sf::cache {

namespace {

CacheDecorator& DecoratorStorage() {
  static CacheDecorator decorator;
  return decorator;
}

} // namespace

void RegisterCacheDecorator(CacheDecorator decorator) {
  DecoratorStorage() = std::move(decorator);
}

void ClearCacheDecorator() {
  DecoratorStorage() = nullptr;
}

CacheDecorator GetCacheDecorator() {
  return DecoratorStorage();
}

} // namespace sf::cache

