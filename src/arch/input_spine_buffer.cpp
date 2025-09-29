#include "arch/input_spine_buffer.hpp"
#include <cstring>
#include <stdexcept>
#include <vector>           // for DRAM fetch buffer
#include "core/clock.hpp"   // to call core fetcher & read batch cursor

namespace sf {

bool InputSpineBuffer::run() {
  // Require a registered core (for batch cursor and DRAM fetcher).
  if (!core_) return false;

  const int batches = core_->batches_needed();
  const int cursor  = core_->load_batch_cursor();
  if (cursor < 0 || cursor >= batches) return false;

  // If we just moved to a new batch, reset all lanes to a clean state.
  if (batch_seen_ != cursor) {
    Flush();
    batch_seen_ = cursor;
  }

  // Policy: load at most ONE segment this call (bandwidth model).
  // Only swap to active if that spine's active bank is empty
  // to avoid overwriting data being drained by Stage-4.
  for (int s = 0; s < kNumSpines; ++s) {
    // If active already has data, let Stage-4 drain it first.
    if (!lanes_[s].active.empty()) continue;

    // Ask core for the next DRAM line (segment) for (batch,cursor,spine=s).
    // The core provides both the bytes and the DramFormat to parse them.
    const sf::dram::DramFormat* fmt = nullptr;
    std::vector<std::uint8_t>   line;
    if (!core_->FetchNextSpineSegment(cursor, s, line, fmt)) {
      // No segment available for this spine right now; try next spine.
      continue;
    }

    if (!fmt || line.empty()) {
      // Malformed provider; ignore and try next spine.
      continue;
    }

    // Load into SHADOW and immediately swap to ACTIVE (since ACTIVE is empty).
    LoadShadowSegmentFromDRAM(s, *fmt, line.data());
    (void)SwapToShadow(s);

    // Loaded one segment this cycle.
    return true;
  }

  // Nothing loaded this call.
  return false;
}

// -------- ctor / reset --------

InputSpineBuffer::InputSpineBuffer() { Flush(); }

void InputSpineBuffer::Flush() {
  for (auto& l : lanes_) {
    l.active.clear();
    l.shadow.clear();
    l.meta = SpineMeta{};
  }
}

// -------- lane binding --------

void InputSpineBuffer::BindLaneIfFirst(int spine_idx, const SegmentHeader& hdr) {
  if (spine_idx < 0 || spine_idx >= kNumSpines) throw std::out_of_range("BindLaneIfFirst: spine_idx");
  auto& m = lanes_[spine_idx].meta;
  if (m.seg_loaded_count == 0) {
    m.batch_id           = hdr.batch_id;
    m.logical_spine_id   = hdr.logical_spine_id;
    m.seg_expected_total = hdr.seg_count;
  }
}

// -------- copy helpers --------

void InputSpineBuffer::copy_entries(Bank& dst, const Entry* src, std::size_t count) {
  if (!src && count) throw std::invalid_argument("copy_entries: null src");
  if (count > LaneCapacity::value) throw std::length_error("copy_entries: exceeds bank capacity");
  if (count) std::memcpy(dst.data.data(), src, count * sizeof(Entry));
}

void InputSpineBuffer::copy_entries_from_raw(Bank& dst, const std::uint8_t* raw, std::size_t count) {
  if (!raw && count) throw std::invalid_argument("copy_entries_from_raw: null raw");
  if (count > LaneCapacity::value) throw std::length_error("copy_entries_from_raw: exceeds bank capacity");
  for (std::size_t i = 0; i < count; ++i) {
    Entry tmp{};
    std::memcpy(&tmp, raw + i * sizeof(Entry), sizeof(Entry));
    dst.data[i] = tmp;
  }
}

// -------- meta update --------

void InputSpineBuffer::update_meta_on_segment(int spine_idx, const SegmentHeader& hdr) {
  auto& meta = lanes_[spine_idx].meta;
  if (meta.seg_loaded_count == 0) {
    meta.batch_id           = hdr.batch_id;
    meta.logical_spine_id   = hdr.logical_spine_id;
    meta.seg_expected_total = hdr.seg_count;
  }
  if (meta.seg_loaded_count < 0xFF) ++meta.seg_loaded_count;
  if (hdr.eol || (meta.seg_expected_total > 0 && meta.seg_loaded_count >= meta.seg_expected_total)) {
    meta.fully_loaded = 1;
  }
}

// -------- load from typed entries --------

void InputSpineBuffer::LoadShadowSegment(int spine_idx, const SegmentHeader& hdr,
                                         const Entry* src, std::size_t count) {
  if (spine_idx < 0 || spine_idx >= kNumSpines) throw std::out_of_range("LoadShadowSegment: spine_idx");
  if (count != static_cast<std::size_t>(hdr.size))
    throw std::invalid_argument("LoadShadowSegment: count must match hdr.size");
  if (count > LaneCapacity::value)
    throw std::length_error("LoadShadowSegment: exceeds bank capacity");

  auto& l  = lanes_[spine_idx];
  auto& sh = l.shadow;

  BindLaneIfFirst(spine_idx, hdr);
  update_meta_on_segment(spine_idx, hdr);

  sh.clear();
  if (count) copy_entries(sh, src, count);
  sh.size = static_cast<uint16_t>(count);
  sh.tail = sh.size;
}

// -------- load from DRAM using format --------

void InputSpineBuffer::LoadShadowSegmentFromDRAM(int spine_idx,
                                                 const DramFormat&   fmt,
                                                 const std::uint8_t* line_base) {
  if (spine_idx < 0 || spine_idx >= kNumSpines) throw std::out_of_range("LoadShadowSegmentFromDRAM: spine_idx");
  if (!line_base) throw std::invalid_argument("LoadShadowSegmentFromDRAM: null line_base");

  // 1) Parse header
  SegmentHeader hdr{};
  fmt.parse_header(line_base, hdr);

  // 2) Locate entries area and determine payload size
  const std::uint8_t* entries_raw = fmt.entries_ptr(line_base);
  const std::size_t   payloadB    = fmt.payload_bytes(hdr);

  // 3) Sanity: payload must be a multiple of sizeof(Entry)
  if (payloadB % sizeof(Entry)) {
    throw std::invalid_argument("LoadShadowSegmentFromDRAM: payload not aligned to Entry size");
  }
  const std::size_t count = payloadB / sizeof(Entry);
  if (count != hdr.size) {
    throw std::invalid_argument("LoadShadowSegmentFromDRAM: size mismatch with payload");
  }
  if (count > LaneCapacity::value) {
    throw std::length_error("LoadShadowSegmentFromDRAM: exceeds bank capacity");
  }

  auto& l  = lanes_[spine_idx];
  auto& sh = l.shadow;

  BindLaneIfFirst(spine_idx, hdr);
  update_meta_on_segment(spine_idx, hdr);

  sh.clear();
  if (count) copy_entries_from_raw(sh, entries_raw, count);
  sh.size = static_cast<uint16_t>(count);
  sh.tail = sh.size;

  // Note: caller can compute next line address as:
  // next = line_base + fmt.line_bytes(hdr)
}

// -------- swap / consume --------

bool InputSpineBuffer::SwapToShadow(int spine_idx) {
  if (spine_idx < 0 || spine_idx >= kNumSpines) return false;

  auto& l  = lanes_[spine_idx];
  auto& sh = l.shadow;
  if (sh.empty()) return false;

  std::swap(l.active, l.shadow);
  l.shadow.clear();
  return true;
}

const Entry* InputSpineBuffer::Head(int spine_idx) const {
  if (spine_idx < 0 || spine_idx >= kNumSpines) return nullptr;
  const auto& a = lanes_[spine_idx].active;
  if (a.empty()) return nullptr;
  return &a.data[a.head];
}

bool InputSpineBuffer::PopHead(int spine_idx) {
  if (spine_idx < 0 || spine_idx >= kNumSpines) return false;
  auto& a = lanes_[spine_idx].active;
  if (a.empty()) return false;
  ++a.head;
  return true;
}

bool InputSpineBuffer::Empty(int spine_idx) const {
  if (spine_idx < 0 || spine_idx >= kNumSpines) return true;
  return lanes_[spine_idx].active.empty();
}

uint16_t InputSpineBuffer::Size(int spine_idx) const {
  if (spine_idx < 0 || spine_idx >= kNumSpines) return 0;
  const auto& a = lanes_[spine_idx].active;
  return static_cast<uint16_t>(a.size - a.head);
}

// -------- lifecycle --------

const InputSpineBuffer::SpineMeta& InputSpineBuffer::LaneMeta(int spine_idx) const {
  if (spine_idx < 0 || spine_idx >= kNumSpines) throw std::out_of_range("LaneMeta: spine_idx");
  return lanes_[spine_idx].meta;
}

void InputSpineBuffer::maybe_mark_fully_drained(LaneDB& l) {
  if (l.meta.fully_loaded && l.active.empty() && l.shadow.empty()) {
    l.meta.fully_drained = 1;
  }
}

void InputSpineBuffer::MarkFullyDrainedIfApplicable(int spine_idx) {
  if (spine_idx < 0 || spine_idx >= kNumSpines) throw std::out_of_range("MarkFullyDrainedIfApplicable: spine_idx");
  auto& l = lanes_[spine_idx];
  maybe_mark_fully_drained(l);
}

} // namespace sf
