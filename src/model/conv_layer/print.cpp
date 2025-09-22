#include "model/conv_layer/print.hpp"
#include "common/constants.hpp"
#include <sstream>

namespace sf {
namespace model {

std::string PrintBatchMap(const driver::BatchSpineMap& m) {
  std::ostringstream oss;
  oss << "BatchSpineMap{num_batches=" << m.NumBatches() << "}\n";
  for (int b = 0; b < m.NumBatches(); ++b) {
    oss << "  Batch " << b << ":\n";
    for (int lane = 0; lane < kNumSpines; ++lane) {
      const auto& addrs = m.Get(b, lane);
      if (addrs.empty()) continue;
      oss << "    lane " << lane
          << " -> blocks=" << addrs.size() << " addrs=[";
      for (size_t i = 0; i < addrs.size() && i < 3; ++i) {
        if (i) oss << ", ";
        oss << addrs[i];
      }
      if (addrs.size() > 3) oss << ", ...";
      oss << "]\n";
    }
  }
  return oss.str();
}

} // namespace model
} // namespace sf
