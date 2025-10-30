// All comments are in English.
#include "cache/mapper_iface.h"

#include <memory>
#include <stdexcept>

#include "cache/cache_config.h"

namespace sf::cache {

class DirectModMapper final : public IMapper {
public:
  explicit DirectModMapper(const CacheConfig& cfg)
      : cfg_(cfg),
        S_tile_(cfg.Cin * cfg.KH * cfg.KW) {
    if (S_tile_ <= 0) {
      throw std::invalid_argument("DirectModMapper: invalid tile size.");
    }
  }

  MapOutput Map(const MapInput& input) const override {
    if (input.tile_id < 0 ||
        input.cin < 0 || input.cin >= cfg_.Cin ||
        input.kh < 0 || input.kh >= cfg_.KH ||
        input.kw < 0 || input.kw >= cfg_.KW) {
      throw std::out_of_range("DirectModMapper::Map: coordinates out of range.");
    }
    const long long L_ll =
        (static_cast<long long>(input.cin) * cfg_.KH + static_cast<long long>(input.kh)) * cfg_.KW +
        static_cast<long long>(input.kw);
    if (L_ll < 0 || L_ll >= S_tile_) {
      throw std::out_of_range("DirectModMapper::Map: L out of range.");
    }
    const int L = static_cast<int>(L_ll);
    const std::uint64_t tag =
        static_cast<std::uint64_t>(static_cast<long long>(input.tile_id) * S_tile_) +
        static_cast<std::uint64_t>(L);
    const int set_idx = L % cfg_.geometry.num_sets;
    return MapOutput{tag, set_idx, L};
  }

private:
  CacheConfig cfg_;
  int S_tile_ = 0;
};

std::unique_ptr<IMapper> MakeDirectModMapper(const CacheConfig& cfg) {
  return std::make_unique<DirectModMapper>(cfg);
}

} // namespace sf::cache

