#pragma once
// All comments are in English.

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "common/constants.hpp"
#include "common/entry.hpp"
#include "arch/intermediate_fifo.hpp"

namespace sf {
  class InputSpineBuffer; // forward declaration; must provide PopSmallestTsEntry(Entry&)
}

/**
 * MinFinderBatch
 *
 * Responsibility:
 *   - Select ONE globally-smallest timestamp Entry via InputSpineBuffer.
 *   - Push it into the IntermediateFIFO chosen by the current batch cursor.
 *   - Track whether the first entry of the last batch was pushed (flag).
 *
 * Wiring:
 *   - Non-owning pointers to InputSpineBuffer and an ARRAY of IntermediateFIFO.
 *   - The array length is defined by kNumIntermediateFifos in common/constants.hpp.
 */
namespace sf {

class MinFinderBatch {
public:
  MinFinderBatch(InputSpineBuffer* isb_ptr,
                 IntermediateFIFO* fifos_array_ptr)
  : isb(isb_ptr),
    fifos(fifos_array_ptr) {}

  // Step once:
  // - Returns true if one entry was successfully pushed into the target FIFO.
  // - Returns false if ISB had no entry or the target FIFO is full (no progress).
  // - Throws std::runtime_error on invalid wiring or out-of-range cursor.
  bool run(int current_batch_cursor, int batches_needed);

  // Query if global merger is allowed to work.
  // Returns true iff the first entry of the LAST batch has been pushed.
  bool CanGlobalMegerWork() const { return last_batch_first_entry_pushed; }

public:
  // Non-owning pointers (required by spec).
  InputSpineBuffer*   isb   = nullptr;                 // (1) pointer to input_spine_buffer
  IntermediateFIFO*   fifos = nullptr;                 // (2) pointer to ARRAY of IntermediateFIFO

  // Internal state.
  Entry picked_entry{};                                 // (3) entry to receive picked/pop result
  bool  last_batch_first_entry_pushed = false;          // (4) tracking flag
  int entry_count_total = 0;                             // (5) total entries pushed (for debug)
};

} // namespace sf
