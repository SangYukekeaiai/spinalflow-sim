// All comments are in English.
#include "cache/prefetch_iface.h"

#include <memory>

namespace sf::cache {

class NoPrefetch final : public IPrefetch {
public:
  PrefetchPlan Plan(const PrefetchInput&) const override {
    return {};
  }
};

std::unique_ptr<IPrefetch> MakeNoPrefetch() {
  return std::make_unique<NoPrefetch>();
}

} // namespace sf::cache

