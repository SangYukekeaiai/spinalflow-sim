#include "model/conv_layer/address.hpp"
#include <algorithm>

namespace sf {
namespace model {

std::vector<driver::BatchSpineMap::Addr>
LogicalSpineAllBlockAddrs_0Based(int s0, int totalSpines, int lenEntries) {
  std::vector<driver::BatchSpineMap::Addr> addrs;
  if (s0 < 0 || s0 >= totalSpines || lenEntries <= 0) return addrs;

  const std::size_t blockBytes = SpineBlockBytes();
  const std::size_t layerBytes = static_cast<std::size_t>(totalSpines) * blockBytes;

  const int numBlocks = (lenEntries + kCapacityPerSpine - 1) / kCapacityPerSpine; // ceil
  addrs.reserve(numBlocks);
  for (int i = 0; i < numBlocks; ++i) {
    const std::uint64_t addr =
        static_cast<std::uint64_t>(i) * layerBytes +
        static_cast<std::uint64_t>(s0) * blockBytes;
    addrs.push_back(static_cast<driver::BatchSpineMap::Addr>(addr));
  }
  return addrs;
}

} // namespace model
} // namespace sf
