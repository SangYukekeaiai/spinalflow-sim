#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "common/constants.hpp"
#include "common/entry.hpp"

namespace sf {

/**
 * InputSpineBuffer
 *
 * Double-buffered per-physical-spine storage. Each physical spine owns two
 * banks: ACTIVE (consumed by downstream) and SHADOW (filled by CPU/driver).
 * SwapToShadow() atomically flips banks when SHADOW has valid data.
 */
class InputSpineBuffer {
public:
  InputSpineBuffer();

  // Remove all entries from all spines and reset indices on both banks.
  void Flush();

  // --- Double-buffer API ---
  // Load entries into the shadow bank of one spine. After SwapToShadow, the
  // shadow content becomes ACTIVE in O(1).
  void LoadSpineShadow(int spine_idx, const Entry* src, std::size_t count);

  // Load from raw bytes (each Entry is expected to be 2 bytes).
  void LoadSpineShadowFromDRAM(int spine_idx, const std::uint8_t* raw_bytes, std::size_t byte_count);

  // Swap active/shadow for a spine (O(1)). Returns false if shadow is empty.
  bool SwapToShadow(int spine_idx);

  // Head/pop on ACTIVE bank only (consumed by MinFinder).
  const Entry* Head(int spine_idx) const;
  bool         PopHead(int spine_idx);

  bool     Empty(int spine_idx) const;
  uint16_t Size(int spine_idx)  const;

private:
  using Lane = std::array<Entry, kCapacityPerSpine>;

  struct Bank {
    Lane     data{};
    uint16_t head = 0;
    uint16_t tail = 0; // size = tail - head
  };

  struct LaneDB {
    Bank active;
    Bank shadow;
  };

  std::array<LaneDB, kNumSpines> lanes_;

  static void copy_from_raw(Lane& dst, const std::uint8_t* raw, std::size_t count);
};

} // namespace sf
