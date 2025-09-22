#include "arch/input_spine_buffer.hpp"

namespace sf {

InputSpineBuffer::InputSpineBuffer() { Flush(); }

void InputSpineBuffer::Flush() {
  for (auto& l : lanes_) {
    l.active.head = l.active.tail = 0;
    l.shadow.head = l.shadow.tail = 0;
  }
}

void InputSpineBuffer::copy_from_raw(Lane& dst, const std::uint8_t* raw, std::size_t count) {
  // We expect Entry to be exactly 2 bytes (ts, neuron_id), both uint8_t.
  static_assert(sizeof(Entry) == 2, "Entry must be 2 bytes");
  for (std::size_t i = 0; i < count; ++i) {
    dst[i].ts        = raw[2 * i + 0];
    dst[i].neuron_id = raw[2 * i + 1];
  }
}

void InputSpineBuffer::LoadSpineShadow(int spine_idx, const Entry* src, std::size_t count) {
  if (spine_idx < 0 || spine_idx >= kNumSpines) throw std::out_of_range("spine_idx");
  if (count > static_cast<std::size_t>(kCapacityPerSpine)) throw std::length_error("exceeds capacity");

  auto& sh = lanes_[spine_idx].shadow;
  sh.head  = 0;
  sh.tail  = static_cast<uint16_t>(count);

  if (count) {
    std::memcpy(sh.data.data(), src, count * sizeof(Entry));
  }
}

void InputSpineBuffer::LoadSpineShadowFromDRAM(int spine_idx, const std::uint8_t* raw_bytes, std::size_t byte_count) {
  if (spine_idx < 0 || spine_idx >= kNumSpines) throw std::out_of_range("spine_idx");
  if (!raw_bytes && byte_count) throw std::invalid_argument("null raw");
  if (byte_count % sizeof(Entry)) throw std::invalid_argument("bad byte count");

  const std::size_t count = byte_count / sizeof(Entry);
  if (count > static_cast<std::size_t>(kCapacityPerSpine)) throw std::length_error("exceeds capacity");

  auto& sh = lanes_[spine_idx].shadow;
  sh.head  = 0;
  sh.tail  = static_cast<uint16_t>(count);
  copy_from_raw(sh.data, raw_bytes, count);
}

bool InputSpineBuffer::SwapToShadow(int spine_idx) {
  if (spine_idx < 0 || spine_idx >= kNumSpines) return false;

  auto& l = lanes_[spine_idx];
  if (l.shadow.head >= l.shadow.tail) return false; // shadow empty => nothing to swap

  std::swap(l.active, l.shadow);
  l.shadow.head = l.shadow.tail = 0; // clear old shadow so it can be refilled
  return true;
}

const Entry* InputSpineBuffer::Head(int spine_idx) const {
  if (spine_idx < 0 || spine_idx >= kNumSpines) return nullptr;
  const auto& a = lanes_[spine_idx].active;
  if (a.head >= a.tail) return nullptr;
  return &a.data[a.head];
}

bool InputSpineBuffer::PopHead(int spine_idx) {
  if (spine_idx < 0 || spine_idx >= kNumSpines) return false;
  auto& a = lanes_[spine_idx].active;
  if (a.head >= a.tail) return false;
  ++a.head;
  return true;
}

bool InputSpineBuffer::Empty(int spine_idx) const {
  if (spine_idx < 0 || spine_idx >= kNumSpines) return true;
  const auto& a = lanes_[spine_idx].active;
  return a.head >= a.tail;
}

uint16_t InputSpineBuffer::Size(int spine_idx) const {
  if (spine_idx < 0 || spine_idx >= kNumSpines) return 0;
  const auto& a = lanes_[spine_idx].active;
  return static_cast<uint16_t>(a.tail - a.head);
}

} // namespace sf
