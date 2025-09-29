#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>

#include "common/constants.hpp"
#include "common/entry.hpp"
#include "arch/dram/dram_common.hpp"
#include "arch/dram/dram_format.hpp"

namespace sf {


class ClockCore;
/**
 * InputSpineBuffer (PSB)
 * Double-buffered per-physical-spine storage with per-spine metadata.
 * Now the PSB can load a segment from DRAM via a pluggable DramFormat.
 */
class InputSpineBuffer {
public:
  // DRAM header mirrored locally
  using SegmentHeader = sf::dram::SegmentHeader;
  using DramFormat    = sf::dram::DramFormat;

  struct SpineMeta {
    // All comments in English
    uint16_t batch_id         = 0;
    uint16_t logical_spine_id = 0;
    uint8_t  seg_expected_total = 0;
    uint8_t  seg_loaded_count   = 0;
    uint8_t  fully_loaded  = 0;
    uint8_t  fully_drained = 0;
  };

  InputSpineBuffer();

  void Flush();

  // Provide core access (for batch cursor, batches_needed, and DRAM fetcher).
  void RegisterCore(ClockCore* core) { core_ = core; }

  // Load at most one segment for the current batch; if active of that spine
  // is empty, swap the newly loaded shadow to active. Returns true if loaded.
  bool run();

  void BindLaneIfFirst(int spine_idx, const SegmentHeader& hdr);

  // Load one segment into SHADOW from typed entries
  void LoadShadowSegment(int spine_idx, const SegmentHeader& hdr,
                         const Entry* src, std::size_t count);

  // Load one segment into SHADOW from a DRAM line buffer using a DramFormat.
  // line_base points at the beginning of the line (header at offset 0).
  void LoadShadowSegmentFromDRAM(int spine_idx,
                                 const DramFormat&     fmt,
                                 const std::uint8_t*   line_base);

  bool SwapToShadow(int spine_idx);

  const Entry* Head(int spine_idx) const;
  bool         PopHead(int spine_idx);

  bool     Empty(int spine_idx) const;
  uint16_t Size(int spine_idx)  const;

  const SpineMeta& LaneMeta(int spine_idx) const;
  void             MarkFullyDrainedIfApplicable(int spine_idx);

private:
  using LaneCapacity = std::integral_constant<std::size_t, kCapacityPerSpine>;

  struct Bank {
    // All comments in English
    std::array<Entry, LaneCapacity::value> data{};
    uint16_t head = 0;
    uint16_t tail = 0;
    uint16_t size = 0;

    inline bool empty() const { return head >= size; }
    inline void clear() { head = tail = size = 0; }
  };

  struct LaneDB {
    Bank      active;
    Bank      shadow;
    SpineMeta meta;
  };

  std::array<LaneDB, kNumSpines> lanes_;

  static void copy_entries(Bank& dst, const Entry* src, std::size_t count);
  static void copy_entries_from_raw(Bank& dst, const std::uint8_t* raw, std::size_t count);

  void update_meta_on_segment(int spine_idx, const SegmentHeader& hdr);
  void maybe_mark_fully_drained(LaneDB& l);
private:
  // ---- New: stage-5 helpers ----
  ClockCore* core_ = nullptr;     // not owned
  int        batch_seen_ = -1;    // detect batch cursor change to reset lanes
};

} // namespace sf
