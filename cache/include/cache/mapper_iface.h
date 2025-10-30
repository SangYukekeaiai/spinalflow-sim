// All comments are in English.
#pragma once

#include <cstdint>

namespace sf::cache {

struct MapInput {
  int tile_id = -1;
  int cin = -1;
  int kh = -1;
  int kw = -1;
};

struct MapOutput {
  std::uint64_t tag = 0;
  int set_idx = -1;
  int L = -1;
};

class IMapper {
public:
  virtual ~IMapper() = default;
  virtual MapOutput Map(const MapInput& input) const = 0;
};

} // namespace sf::cache

