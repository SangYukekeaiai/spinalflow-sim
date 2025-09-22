#pragma once
#include <vector>
#include <stdexcept>
#include "arch/driver/batch_spine_map.hpp"

namespace sf::driver {

// Return DRAM base address (layer-1) for logical spine s in [1..100].
// Each logical spine reserves a 256-byte block (128 entries * 2 bytes).
inline BatchSpineMap::Addr AddrOfLogicalSpineL1(int s) {
  constexpr std::size_t kBlockSize = 256;
  return static_cast<BatchSpineMap::Addr>((s - 1) * kBlockSize);
}

/**
 * Build a 4-batch mapping for a single 7x7 window on a 10x10 grid.
 *
 * Grid: 10x10, row-major logical spine id s = y*10 + x + 1 (x,y: 0-based).
 * Kernel: 7x7, valid top-left positions: ox,oy in [0..3].
 * Enumerate the 49 logical spines inside the window in row-major order,
 * then split into four batches: 16, 16, 16, and 1 element(s).
 *
 * Each physical lane i (0..15) in batch b holds exactly one logical spine's DRAM base address.
 * Unused lanes in the last batch remain empty.
 */
inline BatchSpineMap MakeBatchSpineMapForWindow(int ox, int oy) {
  constexpr int imgW = 10, imgH = 10;
  constexpr int kW = 7, kH = 7;

  if (ox < 0 || oy < 0 || ox + kW > imgW || oy + kH > imgH) {
    throw std::out_of_range("MakeBatchSpineMapForWindow: invalid (ox, oy)");
  }

  // Collect 49 logical spine IDs (row-major within the 7x7 window)
  std::vector<int> logicals;
  logicals.reserve(kW * kH);
  for (int dy = 0; dy < kH; ++dy) {
    for (int dx = 0; dx < kW; ++dx) {
      const int x = ox + dx;
      const int y = oy + dy;
      const int s = y * imgW + x + 1; // s in [1..100]
      logicals.push_back(s);
    }
  }

  // Prepare 4 batches
  BatchSpineMap m(4);

  auto fill_batch = [&](int batchIdx, int start, int count) {
    for (int i = 0; i < count; ++i) {
      const int lane = i; // lanes 0..(count-1)
      const int s    = logicals[start + i];
      m.Set(batchIdx, lane, std::vector<BatchSpineMap::Addr>{AddrOfLogicalSpineL1(s)});
    }
  };

  // 49 -> 16/16/16/1
  fill_batch(/*batchIdx=*/0, /*start=*/0,  /*count=*/16); // indices [0..15]
  fill_batch(/*batchIdx=*/1, /*start=*/16, /*count=*/16); // [16..31]
  fill_batch(/*batchIdx=*/2, /*start=*/32, /*count=*/16); // [32..47]
  fill_batch(/*batchIdx=*/3, /*start=*/48, /*count=*/1 ); // [48] only

  return m;
}

/**
 * Build mappings for all 16 valid windows (oy=0..3, ox=0..3) in row-major order.
 */
inline std::vector<BatchSpineMap> MakeAllWindowsBatchMaps() {
  std::vector<BatchSpineMap> maps;
  maps.reserve(16);
  for (int oy = 0; oy < 4; ++oy) {
    for (int ox = 0; ox < 4; ++ox) {
      maps.push_back(MakeBatchSpineMapForWindow(ox, oy));
    }
  }
  return maps;
}

} // namespace sf::driver
