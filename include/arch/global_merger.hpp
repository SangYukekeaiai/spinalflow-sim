#pragma once
// All comments are in English.

#include <cstddef>
#include <stdexcept>
#include <optional>

#include "common/constants.hpp"
#include "common/entry.hpp"
#include "arch/intermediate_fifo.hpp"
#include "arch/min_finder_batch.hpp"

namespace sf {

/**
 * GlobalMerger
 *
 * Responsibility:
 *   - Check readiness via MinFinderBatch::CanGlobalMegerWork().
 *   - Iterate through the IntermediateFIFOs that belong to MinFinderBatch,
 *     pick the smallest-timestamp head entry (tie-break by neuron_id), then pop it.
 *
 * Wiring:
 *   - Non-owning pointer to the contiguous ARRAY of IntermediateFIFO (size = kNumIntermediateFifos).
 *   - Non-owning reference to MinFinderBatch (to call CanGlobalMegerWork()).
 */
class GlobalMerger {
public:
  // 'fifos_ptr' must point to the same array used by MinFinderBatch.
  GlobalMerger(IntermediateFIFO* fifos_ptr,
               MinFinderBatch& mfb)
  : fifos_(fifos_ptr), mfb_(mfb) {}

  // Run one step:
  //   - Returns true and writes 'out' if an entry was popped from some FIFO.
  //   - Returns false if GlobalMerger is not allowed to work yet, or all FIFOs are empty.
  //   - Throws std::runtime_error if invariants are violated.
  bool run(Entry& out);

private:
  IntermediateFIFO* fifos_ = nullptr; // pointer to the FIFO array (non-owning)
  MinFinderBatch&   mfb_;             // reference to MinFinderBatch (non-owning)
};

} // namespace sf
