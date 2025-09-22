#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>
#include <array>

#include "arch/driver/batch_spine_map.hpp"
#include "arch/driver/make_batch_map_sliding.hpp"

// ---------- Helpers (for expectation checking) ----------

// Recompute the 49 logical spine ids for (ox, oy) window, row-major.
static std::vector<int> ExpectedLogicalsForWindow(int ox, int oy) {
  constexpr int imgW = 10, imgH = 10;
  constexpr int kW = 7, kH = 7;
  (void)imgH; // unused but kept for clarity

  std::vector<int> ids;
  ids.reserve(kW * kH);
  for (int dy = 0; dy < kH; ++dy) {
    for (int dx = 0; dx < kW; ++dx) {
      const int x = ox + dx;
      const int y = oy + dy;
      const int s = y * imgW + x + 1;
      ids.push_back(s);
    }
  }
  return ids;
}

// Pretty print one window's batch mapping in a human-readable way.
static void PrintWindowMap(int ox, int oy, const sf::driver::BatchSpineMap& m) {
  std::cout << "Window (ox=" << ox << ", oy=" << oy << "):\n";
  for (int b = 0; b < 4; ++b) {
    std::cout << "  Batch " << b << ":\n";
    for (int lane = 0; lane < 16; ++lane) {
      const auto& addrs = m.Get(b, lane);
      if (addrs.empty()) continue;
      // Recover logical spine id from address: s = addr/256 + 1
      const auto addr = addrs[0];
      const int s = static_cast<int>(addr / 256) + 1;
      std::cout << "    lane " << lane
                << " -> logical s=" << s
                << " (DRAM= " << addr << ")\n";
    }
  }
}

// Check that the mapping splits the 49 logical spines into 16/16/16/1 and preserves row-major order.
static void CheckWindowMapExpectation(int ox, int oy, const sf::driver::BatchSpineMap& m) {
  auto expected = ExpectedLogicalsForWindow(ox, oy);

  // Gather back from map in the same order: batch0 lanes0..15, batch1 lanes0..15, batch2 lanes0..15, batch3 lane0.
  std::vector<int> reconstructed;
  reconstructed.reserve(49);

  auto harvest_batch = [&](int batchIdx, int want) {
    int taken = 0;
    for (int lane = 0; lane < 16 && taken < want; ++lane) {
      const auto& addrs = m.Get(batchIdx, lane);
      if (addrs.empty()) continue;
      const auto addr = addrs[0];
      const int s = static_cast<int>(addr / 256) + 1;
      reconstructed.push_back(s);
      ++taken;
    }
    assert(taken == want);
  };

  harvest_batch(0, 16);
  harvest_batch(1, 16);
  harvest_batch(2, 16);
  harvest_batch(3, 1);

  // Must match the 49 ids in row-major
  assert(reconstructed.size() == expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    if (reconstructed[i] != expected[i]) {
      std::cerr << "Mismatch at i=" << i
                << " reconstructed=" << reconstructed[i]
                << " expected=" << expected[i] << "\n";
      assert(false && "Batch mapping order mismatch");
    }
  }
}

// ------------------------------ Main test ------------------------------

int main() {
  // Test a single window (e.g., top-left at ox=1, oy=2) and also sweep all windows.

  // Single case
  {
    const int ox = 0, oy = 0;
    auto m = sf::driver::MakeBatchSpineMapForWindow(ox, oy);
    PrintWindowMap(ox, oy, m);
    CheckWindowMapExpectation(ox, oy, m);
  }

  // Full sweep of all valid windows (ox,oy in [0..3])
  {
    auto maps = sf::driver::MakeAllWindowsBatchMaps();
    int idx = 0;
    for (int oy = 0; oy < 4; ++oy) {
      for (int ox = 0; ox < 4; ++ox) {
        const auto& m = maps[idx++];
        CheckWindowMapExpectation(ox, oy, m);
      }
    }
  }

  std::cout << "[OK] Batch mapping format & expectation checks passed.\n";
  return 0;
}
