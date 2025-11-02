// All comments are in English.
#include "cache/registry.h"

#include "cache/cache_config.h"

namespace sf::cache {

std::unique_ptr<IMapper> MakeXorFoldMapper(const CacheConfig& cfg);
std::unique_ptr<IMapper> MakeDirectModMapper(const CacheConfig& cfg);
std::unique_ptr<IReplacement> MakeLruReplacement();
std::unique_ptr<IReplacement> MakeRandomReplacement();
std::unique_ptr<IPrefetch> MakeNoPrefetch();
std::unique_ptr<IWindow> MakeTwoTileWindow();
std::unique_ptr<IWindow> MakeThreeTileWindow();

CacheModules MakeDefaultModules(const CacheConfig& cfg) {
  CacheModules modules;
  modules.mapper = MakeXorFoldMapper(cfg);
  switch (cfg.replacement_kind) {
    case ReplacementKind::Random:
      modules.replacement = MakeRandomReplacement();
      break;
    default:
      modules.replacement = MakeLruReplacement();
      break;
  }
  modules.prefetch = MakeNoPrefetch();
  modules.window = MakeTwoTileWindow();
  return modules;
}

} // namespace sf::cache
