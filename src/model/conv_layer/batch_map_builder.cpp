#include "model/conv_layer/batch_map_builder.hpp"
#include "model/conv_layer/address.hpp"
#include <stdexcept>
#include <algorithm>

namespace sf {
namespace model {

driver::BatchSpineMap BuildBatchMapForWindow(
    int ox, int oy,
    int kW, int kH,
    int inW, int inH,
    int strideW, int strideH,
    int padW, int padH,
    const std::vector<int>& spineLenEntries) {

  if (kW <= 0 || kH <= 0 || inW <= 0 || inH <= 0 ||
      strideW <= 0 || strideH <= 0 || padW < 0 || padH < 0) {
    throw std::invalid_argument("BuildBatchMapForWindow: invalid sizes/stride/padding");
  }
  if (static_cast<int>(spineLenEntries.size()) != inW * inH) {
    throw std::invalid_argument("BuildBatchMapForWindow: spineLenEntries size mismatch");
  }

  const int outW = ((inW + 2*padW - kW) >= 0) ? ((inW + 2*padW - kW) / strideW + 1) : 0;
  const int outH = ((inH + 2*padH - kH) >= 0) ? ((inH + 2*padH - kH) / strideH + 1) : 0;

  if (ox < 0 || oy < 0 || ox >= outW || oy >= outH) {
    throw std::out_of_range("BuildBatchMapForWindow: (ox,oy) out of output range");
  }

  // Collect valid zero-based logical spines within the window.
  std::vector<int> s0_list;
  s0_list.reserve(kW * kH);

  const int x0 = ox * strideW - padW;
  const int y0 = oy * strideH - padH;

  for (int dy = 0; dy < kH; ++dy) {
    for (int dx = 0; dx < kW; ++dx) {
      const int x_in = x0 + dx;
      const int y_in = y0 + dy;
      if (x_in < 0 || x_in >= inW || y_in < 0 || y_in >= inH) continue; // padding
      const int s0 = y_in * inW + x_in; // zero-based logical spine
      s0_list.push_back(s0);
    }
  }

  const int totalTaps = static_cast<int>(s0_list.size());
  const int numBatches = (totalTaps + kNumSpines - 1) / kNumSpines; // ceil
  if (numBatches > kMaxBatches) {
    throw std::runtime_error("BuildBatchMapForWindow: batches exceed kMaxBatches");
  }

  driver::BatchSpineMap map(std::max(1, numBatches));
  int cursor = 0;
  const int totalSpines = inW * inH;

  for (int b = 0; b < numBatches; ++b) {
    const int remaining = totalTaps - cursor;
    const int this_count = std::min(kNumSpines, remaining);
    for (int lane = 0; lane < this_count; ++lane) {
      const int s0 = s0_list[cursor + lane];
      const int len = spineLenEntries[s0]; // entries for this logical spine
      auto addrs = LogicalSpineAllBlockAddrs_0Based(s0, totalSpines, len);
      map.Set(b, lane, addrs); // lane holds multiple DRAM block addrs of this s0
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

} // namespace model
} // namespace sf
