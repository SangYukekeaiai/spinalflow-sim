#include "model/conv_layer.hpp"

#include <sstream>
#include <algorithm>
#include <cmath>

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

driver::BatchSpineMap BuildBatchMapForWindow(
    int ox, int oy,
    int kW, int kH,
    int inW, int inH,
    int strideW, int strideH,
    int padW, int padH,
    const std::vector<int>& spineLenEntries) {

  // ----- Validate sizes -----
  if (kW <= 0 || kH <= 0 || inW <= 0 || inH <= 0 ||
      strideW <= 0 || strideH <= 0 || padW < 0 || padH < 0) {
    throw std::invalid_argument("BuildBatchMapForWindow: invalid sizes/stride/padding");
  }
  if (static_cast<int>(spineLenEntries.size()) != inW * inH) {
    throw std::invalid_argument("BuildBatchMapForWindow: spineLenEntries size mismatch");
  }

  // Output shape (valid conv with stride & padding)
  const int outW = ((inW + 2*padW - kW) >= 0) ? ((inW + 2*padW - kW) / strideW + 1) : 0;
  const int outH = ((inH + 2*padH - kH) >= 0) ? ((inH + 2*padH - kH) / strideH + 1) : 0;

  if (ox < 0 || oy < 0 || ox >= outW || oy >= outH) {
    throw std::out_of_range("BuildBatchMapForWindow: (ox,oy) out of output range");
  }

  // ----- 1) Collect valid zero-based logical spines within the window -----
  std::vector<int> s0_list;
  s0_list.reserve(kW * kH);

  const int x0 = ox * strideW - padW;
  const int y0 = oy * strideH - padH;

  for (int dy = 0; dy < kH; ++dy) {
    for (int dx = 0; dx < kW; ++dx) {
      const int x_in = x0 + dx;
      const int y_in = y0 + dy;
      if (x_in < 0 || x_in >= inW || y_in < 0 || y_in >= inH) continue; // padding tap
      const int s0 = y_in * inW + x_in; // 0-based
      s0_list.push_back(s0);
    }
  }

  // ----- 2) Split taps into batches of size kNumSpines -----
  const int totalTaps = static_cast<int>(s0_list.size());
  const int numBatches = (totalTaps + kNumSpines - 1) / kNumSpines; // ceil
  if (numBatches > kMaxBatches) {
    throw std::runtime_error("BuildBatchMapForWindow: batches exceed kMaxBatches");
  }

  driver::BatchSpineMap map(std::max(1, numBatches)); // at least 1 batch

  // ----- 3) Fill each batch lane with ALL block addresses for that logical spine -----
  int cursor = 0;
  const int totalSpines = inW * inH;
  for (int b = 0; b < numBatches; ++b) {
    const int remaining = totalTaps - cursor;
    const int this_count = std::min(kNumSpines, remaining);
    for (int lane = 0; lane < this_count; ++lane) {
      const int s0 = s0_list[cursor + lane];
      const int len = spineLenEntries[s0]; // entries for this logical spine
      auto addrs = LogicalSpineAllBlockAddrs_0Based(s0, totalSpines, len);
      // Even if len==0, addrs will be empty; leaving lane empty is acceptable.
      map.Set(b, lane, addrs);
    }
    cursor += this_count;
  }

  return map;
}

std::vector<driver::BatchSpineMap> BuildAllWindowsBatchMaps(
    int kW, int kH,
    int inW, int inH,
    int strideW, int strideH,
    int padW, int padH,
    const std::vector<int>& spineLenEntries) {

  if (kW <= 0 || kH <= 0 || inW <= 0 || inH <= 0 ||
      strideW <= 0 || strideH <= 0 || padW < 0 || padH < 0) {
    throw std::invalid_argument("BuildAllWindowsBatchMaps: invalid sizes/stride/padding");
  }
  if (static_cast<int>(spineLenEntries.size()) != inW * inH) {
    throw std::invalid_argument("BuildAllWindowsBatchMaps: spineLenEntries size mismatch");
  }

  const int outW = ((inW + 2*padW - kW) >= 0) ? ((inW + 2*padW - kW) / strideW + 1) : 0;
  const int outH = ((inH + 2*padH - kH) >= 0) ? ((inH + 2*padH - kH) / strideH + 1) : 0;

  std::vector<driver::BatchSpineMap> maps;
  maps.reserve(std::max(0, outW * outH));

  for (int oy = 0; oy < outH; ++oy) {
    for (int ox = 0; ox < outW; ++ox) {
      maps.push_back(BuildBatchMapForWindow(
          ox, oy, kW, kH, inW, inH, strideW, strideH, padW, padH, spineLenEntries));
    }
  }
  return maps;
}

std::string PrintBatchMap(const driver::BatchSpineMap& m, int inW) {
  const std::size_t blockBytes = SpineBlockBytes();

  std::ostringstream oss;
  oss << "BatchSpineMap{num_batches=" << m.NumBatches() << "}\n";
  for (int b = 0; b < m.NumBatches(); ++b) {
    oss << "  Batch " << b << ":\n";
    for (int lane = 0; lane < kNumSpines; ++lane) {
      const auto& addrs = m.Get(b, lane);
      if (addrs.empty()) continue;

      // Try to infer a representative s0 from the first address
      const std::size_t layerBytes = static_cast<std::size_t>(inW) * /*inH cancels via modulo layerBytes*/ 0; // placeholder
      // Note: exact reverse-mapping s0 requires knowing layerBytes = totalSpines*blockBytes.
      // Here we instead derive s0 by modulo with blockBytes stride along a single layer:
      const driver::BatchSpineMap::Addr a0 = addrs.front();
      const int s0_guess = static_cast<int>((a0 % (static_cast<std::uint64_t>(inW) * 1 /*we don't know inH here*/)) / blockBytes);

      oss << "    lane " << lane
          << " -> blocks=" << addrs.size();

      // Print up to first 3 addresses
      oss << " addrs=[";
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
