#pragma once
#include <cstdint>
#include <unordered_map>
#include <memory>
#include <vector>

#include "arch/dram/stream_reader.hpp"  // DramImage, StreamReader, LayerDirectory, SegmentHeader
#include "arch/dram/dram_format.hpp"    // DramFormat

namespace sf { namespace dram {

/**
 * InputSpineFetcher
 *
 * Bridges StreamReader to ClockCore::SetDramFetcher.
 * Each physical spine gets its own StreamReader instance so that
 * every spine maintains an independent scan cursor inside the layer's
 * inputs range (no interference across spines).
 *
 * Design notes:
 *  - Batch is a runtime scheduling concept → ignored here.
 *  - The directory is batch-agnostic: we open the layer's inputs range.
 *  - 'operator()' matches the Core's fetcher signature and is intended
 *    to be passed directly into ClockCore::SetDramFetcher(...).
 */
class InputSpineFetcher {
public:
  // The fetcher does not own fmt/img/dir; they must outlive this object.
  InputSpineFetcher(const DramFormat&         fmt,
                    const DramImage&          img,
                    const LayerDirectory&     dir,
                    std::uint16_t             layer_id);

  // Matches ClockCore::DramFetchFn signature.
  // Returns true if a segment for the given 'spine' is available.
  // The 'batch' parameter is ignored by design (batch-agnostic directory).
  bool operator()(int /*batch_ignored*/,
                  int spine,
                  std::vector<std::uint8_t>& out_line,
                  const sf::dram::DramFormat*& out_fmt);

  // Optional helpers:
  void   SetLayer(std::uint16_t new_layer); // reset all per-spine readers
  void   ResetAll();                        // drop all cursors (spines)
  bool   Eof(std::uint16_t spine) const;    // true if that spine has no more segments

private:
  // Lazily allocate and open a StreamReader for a spine if not present.
  // Returns nullptr if open_spine(...) fails (e.g., empty inputs range).
  StreamReader* ensure_reader(std::uint16_t spine);

private:
  // Non-owning references to shared resources.
  const DramFormat*     fmt_  = nullptr;
  const DramImage*      img_  = nullptr;
  const LayerDirectory* dir_  = nullptr;

  std::uint16_t layer_id_ = 0;

  // One reader per spine so each keeps an independent cursor.
  std::unordered_map<std::uint16_t, std::unique_ptr<StreamReader>> readers_;

  // Spines for which read_next(...) has reached the end (no more segments).
  std::unordered_map<std::uint16_t, bool> eof_flags_;
};

}} // namespace sf::dram
