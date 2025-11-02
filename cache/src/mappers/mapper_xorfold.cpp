// All comments are in English.
#include "cache/mapper_iface.h"

#include <cstdint>
#include <memory>
#include <stdexcept>

#include "cache/cache_config.h"

namespace sf::cache {

namespace {

inline std::uint32_t mix64(std::uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return static_cast<std::uint32_t>(x);
}

inline std::uint32_t index_xorfold(std::uint64_t line_addr,
                                   std::uint32_t num_sets,
                                   std::uint32_t channel_id) {
  const std::uint64_t folded =
      line_addr ^ (line_addr >> 11) ^
      (static_cast<std::uint64_t>(channel_id) * 0x9e3779b97f4a7c15ULL);
  const std::uint32_t h = mix64(folded);
  if (num_sets == 0u) {
    return 0u;
  }
  return ((num_sets & (num_sets - 1u)) == 0u)
             ? (h & (num_sets - 1u))
             : (h % num_sets);
}

} // namespace

class XorFoldMapper final : public IMapper {
public:
  explicit XorFoldMapper(const CacheConfig& cfg)
      : cfg_(cfg),
        S_tile_(cfg.Cin * cfg.KH * cfg.KW),
        num_sets_(cfg.geometry.num_sets) {
    if (S_tile_ <= 0) {
      throw std::invalid_argument("XorFoldMapper: invalid tile size.");
    }
    if (num_sets_ <= 0) {
      throw std::invalid_argument("XorFoldMapper: num_sets must be positive.");
    }
  }

  MapOutput Map(const MapInput& input) const override {
    if (input.tile_id < 0 ||
        input.cin < 0 || input.cin >= cfg_.Cin ||
        input.kh < 0 || input.kh >= cfg_.KH ||
        input.kw < 0 || input.kw >= cfg_.KW) {
      throw std::out_of_range("XorFoldMapper::Map: coordinates out of range.");
    }

    const long long L_ll =
        (static_cast<long long>(input.cin) * cfg_.KH + static_cast<long long>(input.kh)) * cfg_.KW +
        static_cast<long long>(input.kw);
    if (L_ll < 0 || L_ll >= S_tile_) {
      throw std::out_of_range("XorFoldMapper::Map: L out of range.");
    }
    const int L = static_cast<int>(L_ll);

    const std::uint64_t key =
        static_cast<std::uint64_t>(static_cast<long long>(input.tile_id) * S_tile_) +
        static_cast<std::uint64_t>(L);

    const std::uint32_t set_idx_u32 =
        index_xorfold(key,
                      static_cast<std::uint32_t>(num_sets_),
                      static_cast<std::uint32_t>(input.cin));

    if (set_idx_u32 >= static_cast<std::uint32_t>(num_sets_)) {
      throw std::runtime_error("XorFoldMapper::Map: computed set index out of range.");
    }

    return MapOutput{key, static_cast<int>(set_idx_u32), L};
  }

private:
  CacheConfig cfg_;
  int S_tile_ = 0;
  int num_sets_ = 0;
};

std::unique_ptr<IMapper> MakeXorFoldMapper(const CacheConfig& cfg) {
  return std::make_unique<XorFoldMapper>(cfg);
}

} // namespace sf::cache

