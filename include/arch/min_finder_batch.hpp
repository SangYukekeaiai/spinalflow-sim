#pragma once
// All comments are in English.

#include <optional>
#include "common/constants.hpp"
#include "arch/input_spine_buffer.hpp"
#include "arch/intermediate_fifo.hpp"

namespace sf {

// Forward declare to avoid heavy includes here.
class ClockCore;

/**
 * MinFinderBatch (Stage 4 integration)
 *
 * Selects the minimum timestamp head across all physical spines (ACTIVE bank)
 * and pushes entries into an IntermediateFIFO for the CURRENT batch.
 *
 * New behavior:
 *  - RegisterCore(core) to access batch cursor, FIFOs, and control flags.
 *  - run():
 *      * If cursor >= batches_needed -> stall.
 *      * Let b = cursor; if FIFO[b] is full -> stall.
 *      * Try DrainOneInto(FIFO[b]); if moved -> return true.
 *      * If not moved, check if all spines are empty; if so:
 *          input_drained[b] = true; advance cursor to next batch.
 *        Return false.
 */
class MinFinderBatch {
public:
  explicit MinFinderBatch(InputSpineBuffer& buf) : buf_(buf) {}

  // Stage-4 integration
  void RegisterCore(ClockCore* core) { core_ = core; }
  bool run();

  // Existing APIs (kept for reuse/tests)
  std::size_t DrainBatchInto(IntermediateFIFO& dst);
  bool        DrainOneInto(IntermediateFIFO& dst);

private:
  std::optional<Entry> PopMinHead();

private:
  InputSpineBuffer& buf_;
  ClockCore*        core_ = nullptr; // not owned
};

} // namespace sf
