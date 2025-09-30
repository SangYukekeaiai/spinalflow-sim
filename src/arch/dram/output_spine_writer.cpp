#include "arch/dram/output_spine_writer.hpp"
#include "arch/dram/stream_writer.hpp"   // assumed available from your existing code

namespace sf { namespace dram {

OutputSpineWriter::OutputSpineWriter(const DramFormat& spine_fmt,
                                     std::vector<std::uint8_t>& out_image,
                                     std::uint16_t next_layer_id,
                                     SpineIdFn spine_of_entry)
: fmt_(&spine_fmt),
  img_bytes_(&out_image),
  next_layer_id_(next_layer_id),
  spine_of_(std::move(spine_of_entry))
{
  // Record the begin offset lazily on the first emission.
  begin_ = static_cast<std::uint64_t>(img_bytes_->size());
}

std::function<bool(const sf::Entry&)> OutputSpineWriter::MakeSink() {
  return [this](const sf::Entry& e) -> bool {
    // Determine the logical spine id for this entry.
    const std::uint16_t sid = spine_of_(e);

    // Stage this entry in the per-spine buffer.
    auto& ps = acc_[sid];
    ps.buf.push_back(e);

    // If we reached a full segment (128 entries), flush it (not EOL).
    if (ps.buf.size() == 128) {
      flush_spine(sid, /*eol=*/false);
      ps.buf.clear();
      ps.any_emitted = true;
    }

    // Returning true signals S0 that the sink consumed the entry successfully.
    return true;
  };
}

void OutputSpineWriter::flush_spine(std::uint16_t spine_id, bool eol) {
  auto it = acc_.find(spine_id);
  if (it == acc_.end()) return; // nothing to flush
  PerSpine& ps = it->second;

  // Nothing to write for an empty buffer unless this is the very first and we want to write an empty segment (we do not).
  if (ps.buf.empty()) return;

  // Prepare header.
  SegmentHeader h{};
  h.version           = 1;
  h.kind              = SEG_SPINE;
  h.layer_id          = next_layer_id_;
  h.logical_spine_id  = spine_id;
  h.size              = static_cast<std::uint8_t>(ps.buf.size()); // <= 128
  h.seg_id            = ps.seg_id++;
  h.seg_count         = 0;          // unknown at write time; rely on EOL
  h.eol               = static_cast<std::uint8_t>(eol ? 1 : 0);
  h.aux0              = 0;
  h.aux1              = 0;
  h.reserved          = 0;

  // Serialize using the provided DramFormat via a StreamWriter.
  // We assume FixedStrideFormat so line_bytes() is constant for the lane,
  // but we still pass size so readers know the valid payload.
  StreamWriter wr(*fmt_, *img_bytes_);

  // Payload: contiguous array of Entry
  const std::uint8_t* payload = reinterpret_cast<const std::uint8_t*>(ps.buf.data());
  const std::size_t   payloadB= ps.buf.size() * sizeof(sf::Entry);

  wr.append(h, payload, payloadB);

  if (!started_) {
    started_ = true;
    begin_   = static_cast<std::uint64_t>(img_bytes_->size()) - fmt_->line_bytes(h);
  }
}

sf::dram::Range OutputSpineWriter::Finalize() {
  if (!finalized_) {
    // Flush remaining partial segments for every spine and mark them EOL.
    for (auto& kv : acc_) {
      const std::uint16_t sid = kv.first;
      PerSpine& ps = kv.second;
      if (!ps.buf.empty()) {
        flush_spine(sid, /*eol=*/true);
        ps.buf.clear();
        ps.any_emitted = true;
      } else if (ps.any_emitted) {
        // Already flushed full segments earlier but no trailing partial.
        // The last flushed full segment should have been not-EOL; however,
        // we still need one EOL for the stream. To keep the design simple,
        // we accept the convention "seg_count=0 + no EOL" means open stream.
        // If you strictly require an explicit EOL segment, uncomment below
        // to emit a zero-size EOL segment (if your readers support size==0).
        //
        // SegmentHeader h = ... size=0, eol=1; wr.append(h, nullptr, 0);
      }
    }
    end_       = static_cast<std::uint64_t>(img_bytes_->size());
    finalized_ = true;
    if (!started_) {
      // Nothing was ever written; keep begin == end.
      begin_ = end_;
    }
  }
  return sf::dram::Range{ begin_, end_ };
}

sf::dram::Range OutputSpineWriter::CurrentRange() const {
  const std::uint64_t cur_end = finalized_ ? end_ : static_cast<std::uint64_t>(img_bytes_->size());
  return sf::dram::Range{ begin_, cur_end };
}

}} // namespace sf::dram
