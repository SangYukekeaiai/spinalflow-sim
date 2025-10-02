// arch/output_queue.cpp
#include "arch/output_queue.hpp"

namespace sf {

OutputQueue::OutputQueue(std::size_t capacity_entries)
  : capacity_entries_(capacity_entries) {}

void OutputQueue::clear() {
  staged_.clear();
  partial_.clear();
  ready_lines_.clear();
  total_entries_ = 0;
  active_spine_ = 0;
}

bool OutputQueue::push_entry(const Entry& e) {
  // Capacity check: count this entry as resident until drained to writer.
  if (total_entries_ + 1 > capacity_entries_) return false;
  staged_.push_back(e);
  ++total_entries_;
  return true;
}

OutputQueue::LineBuf& OutputQueue::buf_for(std::uint16_t spine) {
  auto it = partial_.find(spine);
  if (it == partial_.end()) {
    it = partial_.emplace(spine, LineBuf{}).first;
  }
  return it->second;
}

void OutputQueue::seal_full_line(std::uint16_t spine, LineBuf& lb) {
  LinePacket pkt;
  pkt.spine_id = spine;
  pkt.count    = 128;
  pkt.is_full  = true;
  for (std::uint16_t i = 0; i < 128; ++i) pkt.entries[i] = lb.buf[i];
  ready_lines_.push_back(std::move(pkt));
  // We keep total_entries_ unchanged here; it is decremented when drained.
  lb.fill = 0;
}

bool OutputQueue::run() {
  // Always publish S0->S1 valid before doing any work so S1 sees the latest availability.
  // "Valid" here means "S0 is available for downstream to make progress".
  if (core_) core_->SetSt0St1Valid(!full());

  // Ingest all staged entries into the current active spine's partial buffer.
  if (staged_.empty()) {
    // Nothing to ingest this cycle; still return after publishing valid.
    return false;
  }

  bool progressed = false;
  LineBuf& lb = buf_for(active_spine_);

  for (const Entry& e : staged_) {
    // Append into the active spine's line buffer
    lb.buf[lb.fill++] = e;
    progressed = true;

    // When full, seal into a ready full line
    if (lb.fill == 128) {
      seal_full_line(active_spine_, lb);
    }
  }
  staged_.clear();

  // Publish again after mutations, in case we just turned full by ingesting.
  if (core_) core_->SetSt0St1Valid(!full());

  return progressed;
}

void OutputQueue::FlushAllPartialLines() {
  for (auto& kv : partial_) {
    const std::uint16_t spine = kv.first;
    LineBuf& lb = kv.second;
    if (lb.fill == 0) continue;

    LinePacket pkt;
    pkt.spine_id = spine;
    pkt.count    = lb.fill;
    pkt.is_full  = false;
    for (std::uint16_t i = 0; i < lb.fill; ++i) pkt.entries[i] = lb.buf[i];
    ready_lines_.push_back(std::move(pkt));

    lb.fill = 0;
    // Note: total_entries_ will be decremented upon draining ready_lines_.
  }
}

void OutputQueue::DrainAllReadyLines(std::vector<LinePacket>& out) {
  // Move all ready lines into 'out' and update total_entries_ accordingly.
  while (!ready_lines_.empty()) {
    LinePacket pkt = std::move(ready_lines_.front());
    ready_lines_.pop_front();

    if (total_entries_ >= pkt.count) total_entries_ -= pkt.count;
    else total_entries_ = 0;

    out.push_back(std::move(pkt));
  }
}

} // namespace sf
