// All comments are in English.
#include "cache/mapper_iface.h"

#include <limits>
#include <memory>
#include <stdexcept>

#include "cache/cache_config.h"

namespace sf::cache {

namespace {
inline int positive_mod(long long value, int mod) {
  if (mod <= 0) return 0;
  long long res = value % mod;
  if (res < 0) res += mod;
  return static_cast<int>(res);
}
} // namespace

class AffineColorPinMapper final : public IMapper {
public:
  explicit AffineColorPinMapper(const CacheConfig& cfg)
      : cfg_(cfg),
        S_tile_(cfg.Cin * cfg.KH * cfg.KW),
        N_color_(cfg.geometry.num_sets / 2) {
    if (S_tile_ <= 0) {
      throw std::invalid_argument("AffineColorPinMapper: invalid tile size.");
    }
  }

  MapOutput Map(const MapInput& input) const override {
    if (input.tile_id < 0 ||
        input.cin < 0 || input.cin >= cfg_.Cin ||
        input.kh < 0 || input.kh >= cfg_.KH ||
        input.kw < 0 || input.kw >= cfg_.KW) {
      throw std::out_of_range("AffineColorPinMapper::Map: coordinates out of range.");
    }
    const long long L_ll =
        (static_cast<long long>(input.cin) * cfg_.KH + static_cast<long long>(input.kh)) * cfg_.KW +
        static_cast<long long>(input.kw);
    if (L_ll < 0 || L_ll >= S_tile_) {
      throw std::out_of_range("AffineColorPinMapper::Map: L out of range.");
    }
    const int L = static_cast<int>(L_ll);
    const std::uint64_t tag =
        static_cast<std::uint64_t>(static_cast<long long>(input.tile_id) * S_tile_) +
        static_cast<std::uint64_t>(L);

    const int idx_in_color = positive_mod(static_cast<long long>(cfg_.A1) * L, N_color_);
    const int parity = input.tile_id & 1;
    const int set_idx = (idx_in_color << 1) | parity;

    return MapOutput{tag, set_idx, L};
  }

private:
  CacheConfig cfg_;
  int S_tile_ = 0;
  int N_color_ = 0;
};

std::unique_ptr<IMapper> MakeAffineColorPinMapper(const CacheConfig& cfg) {
  return std::make_unique<AffineColorPinMapper>(cfg);
}

} // namespace sf::cache

