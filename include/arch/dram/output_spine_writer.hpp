#pragma once
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <functional>
#include <memory>
#include <cstring>

#include "common/entry.hpp"
#include "arch/dram/dram_common.hpp"     // SegmentHeader, SEG_SPINE
#include "arch/dram/dram_format.hpp"     // DramFormat
#include "arch/dram/stream_reader.hpp"   // DramImage (for size), StreamWriter declaration
#include "arch/dram/layer_directory.hpp" // Range

namespace sf { namespace dram {

/**
 * OutputSpineWriter
 *
 * Collects core outputs (Entry) and packages them into SEG_SPINE segments,
 * 128 entries per segment, written via StreamWriter into a DRAM image buffer.
 *
 * The emitted segments are tagged for the *next* layer (layer_id = next_layer_id).
 * After Finalize(), the [begin,end) range that was written can be used as the
 * LayerDirectory::input_range(next_layer_id).
 *
 * Design:
 *  - Batching is a runtime scheduling concern, not stored in headers.
 *  - We do NOT populate seg_count (set it to 0). We rely on 'eol=1' on the last segment.
 *  - We maintain per-spine seg_id counters (0,1,2,...) as we flush.
 *  - 'logical_spine_id' must be supplied by a policy function (spine_of_entry).
 *  - Payload layout = contiguous array of Entry (sizeof(Entry) * N), N<=128.
 */
class OutputSpineWriter {
public:
  // Map an Entry to its logical spine id (e.g., h*W + w). Must be provided by the caller.
  using SpineIdFn = std::function<std::uint16_t(const sf::Entry&)>;

  // The writer does not own fmt or the output byte buffer; they must outlive this object.
  // 'out_image' is the same vector that holds your DRAM image bytes.
  OutputSpineWriter(const DramFormat&     spine_fmt,
                    std::vector<std::uint8_t>& out_image,
                    std::uint16_t        next_layer_id,
                    SpineIdFn            spine_of_entry);

  // Build a sink compatible with ClockCore::SetOutputSink(...).
  // Pass this to core.SetOutputSink(writer.MakeSink()).
  std::function<bool(const sf::Entry&)> MakeSink();

  // Flush all pending partial segments and mark them as EOL.
  // Returns the [begin,end) range covering everything this writer emitted.
  // Calling Finalize() twice is safe (the second call is a no-op and returns the same range).
  sf::dram::Range Finalize();

  // Range view without forcing finalize; begins at the first emitted byte.
  // Useful if you want to peek where the region starts. End is 0 until Finalize().
  sf::dram::Range CurrentRange() const;

private:
  struct PerSpine {
    std::vector<sf::Entry> buf; // staged entries for the current segment
    std::uint16_t seg_id = 0;   // next segment id to assign upon flush
    bool any_emitted = false;   // whether we've written any segment yet
  };

  // Append a full or partial segment for 'spine_id'.
  // If 'eol' is true, this is the last segment for this spine.
  void flush_spine(std::uint16_t spine_id, bool eol);

private:
  const DramFormat*               fmt_;         // entry_bytes == sizeof(sf::Entry), maxEntries == 128
  std::vector<std::uint8_t>*      img_bytes_;   // output DRAM image buffer (shared with the rest of the image)
  std::uint16_t                   next_layer_id_;
  SpineIdFn                       spine_of_;

  // Per-spine accumulators
  std::unordered_map<std::uint16_t, PerSpine> acc_;

  // Region begin/end (absolute offsets in img_bytes_). 'end_' is set in Finalize().
  std::uint64_t begin_ = 0;
  std::uint64_t end_   = 0;
  bool          started_  = false;
  bool          finalized_= false;
};

}} // namespace sf::dram
