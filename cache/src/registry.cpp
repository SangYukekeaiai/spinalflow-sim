// All comments are in English.
#include "cache/registry.h"

#include "cache/cache_config.h"
#include "cache/belady.h"

namespace sf::cache {

std::unique_ptr<IMapper> MakeXorFoldMapper(const CacheConfig& cfg);
std::unique_ptr<IMapper> MakeDirectModMapper(const CacheConfig& cfg);
std::unique_ptr<IReplacement> MakeLruReplacement();
std::unique_ptr<IReplacement> MakeRandomReplacement();
std::unique_ptr<IReplacement> MakeBeladyReplacement(const CacheConfig& cfg);
std::unique_ptr<IPrefetch> MakeFirstTouchPrefetch(const CacheConfig& cfg);
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
    case ReplacementKind::Belady:
      modules.replacement = MakeBeladyReplacement(cfg);
      break;
    default:
      modules.replacement = MakeLruReplacement();
      break;
  }
  if (cfg.replacement_kind == ReplacementKind::Belady) {
    modules.prefetch = MakeNoPrefetch();
  } else if (cfg.prefetch_buffer_enabled) {
    modules.prefetch = MakeFirstTouchPrefetch(cfg);
  } else {
    modules.prefetch = MakeNoPrefetch();
  }
  modules.window = MakeTwoTileWindow();
  if (cfg.prefetch_buffer_enabled && cfg.replacement_kind != ReplacementKind::Belady) {
    modules.prefetch_buffer = std::make_unique<PrefetchBuffer>(1024);
  }
  return modules;
}

} // namespace sf::cache
