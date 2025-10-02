// arch/output_queue.hpp
#pragma once
#include <vector>
#include <deque>
#include <unordered_map>
#include <cstddef>
#include <cstdint>
#include <functional>
#include "common/constants.hpp"
#include "common/entry.hpp"
#include "core/core_iface.hpp"

namespace sf {

// A full or tail line to be written later to DRAM.
struct LinePacket {
  std::uint16_t spine_id = 0;   // logical spine id (e.g., h*W + w)
  Entry         entries[128];   // payload entries
  std::uint16_t count = 0;      // number of valid entries (<=128)
  bool          is_full = false;// true if count==128
};

class OutputQueue {
public:
  explicit OutputQueue(std::size_t capacity_entries = kDefaultOutputQueueCapacity);

  // Producer API (PEs call this): stage one entry for the current cycle.
  // We do NOT compute spine here; spine is provided via SetActiveSpine().
  bool push_entry(const Entry& e);

  // Stage-0 per-cycle work: ingest all staged entries into the line buffer
  // of the current active spine; seal full lines (128) into ready_lines_.
  // Returns true if any progress (moved entries or sealed a line).
  bool run();

  // The controller (ConvRunner/ClockCore) must set which spine is active
  // for the current position; all staged entries of this cycle go to this spine.
  void SetActiveSpine(std::uint16_t spine_id) { active_spine_ = spine_id; }

  // Flush all partial (non-full) line buffers into ready_lines_ as tail lines.
  void FlushAllPartialLines();

  // Drain all ready lines (full & tail) into 'out' for batch writeback.
  // This does NOT touch partial buffers.
  void DrainAllReadyLines(std::vector<LinePacket>& out);

  // Introspection
  bool   empty() const { return staged_.empty() && ready_lines_.empty() && total_entries_ == 0; }
  size_t total_entries() const { return total_entries_; }
  size_t capacity() const { return capacity_entries_; }

  // Clear everything (use with care).
  void clear();

  // Keep a hook to the core for cohesion (not used for writeback anymore).
  void RegisterCore(CoreIface* core) { core_ = core; }

  bool   empty() const { return staged_.empty() && ready_lines_.empty() && total_entries_ == 0; }
  size_t total_entries() const { return total_entries_; }
  size_t capacity() const { return capacity_entries_; }

  // NEW: fullness query used by SmallestTsPicker
  bool   full() const { return total_entries_ >= capacity_entries_; }

private:
  // Per-spine partial line buffer (<=128 entries).
  struct LineBuf {
    Entry buf[128];
    std::uint16_t fill = 0;
  };

  // Get or create the partial buffer for a spine.
  LineBuf& buf_for(std::uint16_t spine);

  // Seal a full 128-entry line into ready_lines_.
  void seal_full_line(std::uint16_t spine, LineBuf& lb);

private:
  // Capacity counts ALL entries still resident in OutputQueue:
  // staged_ + partial_ (fills) + ready_lines_ (payloads).
  std::size_t capacity_entries_;
  std::size_t total_entries_ = 0;

  // Current active spine id (declared by controller before run()).
  std::uint16_t active_spine_ = 0;

  // Entries produced in this cycle, staged by push_entry(), consumed in run().
  std::vector<Entry> staged_;

  // Per-spine partial line buffers.
  std::unordered_map<std::uint16_t, LineBuf> partial_;

  // Ready-to-write lines (full or flushed tail).
  std::deque<LinePacket> ready_lines_;

  // Core hook (not used for writeback now).
  CoreIface* core_ = nullptr;
};

} // namespace sf
