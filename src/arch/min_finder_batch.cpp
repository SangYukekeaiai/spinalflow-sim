// All comments are in English.

#include "arch/min_finder_batch.hpp"
#include "arch/input_spine_buffer.hpp"  // must provide: bool PopSmallestTsEntry(Entry&)

namespace sf {

bool MinFinderBatch::run(int current_batch_cursor, int batches_needed) {
  // Validate wiring first.
  if (!isb) {
    throw std::runtime_error("MinFinderBatch::run: null InputSpineBuffer pointer.");
  }
  if (!fifos) {
    throw std::runtime_error("MinFinderBatch::run: null IntermediateFIFO array pointer.");
  }
  if (batches_needed <= 0) {
    throw std::runtime_error("MinFinderBatch::run: invalid batches_needed (<= 0).");
  }

  // 1) Select the entry with the smallest timestamp from ISB.
  //    Implementation is delegated to InputSpineBuffer::PopSmallestTsEntry.
  if (!isb->PopSmallestTsEntry(picked_entry)) {
    // ISB has no available entry: nothing to do this cycle.
    return false;
  }

  // 2) Push the selected entry to the related intermediate FIFO:
  //    First check current cursor and index the FIFO array.
  if (current_batch_cursor < 0 ||
      static_cast<std::size_t>(current_batch_cursor) >= kNumIntermediateFifos) {
    // Per your requirement: if this happens, jump out an error.
    throw std::runtime_error("MinFinderBatch::run: current_batch_cursor out of range.");
  }

  IntermediateFIFO& fifo = fifos[static_cast<std::size_t>(current_batch_cursor)];

  // If the target FIFO is full, do not pop/push further this cycle.
  if (fifo.full()) {
    return false;
  }

  // Attempt to push. If push fails unexpectedly (despite not full), treat as invariant violation.
  if (!fifo.push(picked_entry)) {
    throw std::runtime_error("MinFinderBatch::run: FIFO push failed unexpectedly.");
  }

  // 3) Update the flag for "last batch's first entry pushed".
  //    If already true, we leave it unchanged. Otherwise:
  if (!last_batch_first_entry_pushed) {
    if (current_batch_cursor == (batches_needed - 1)) {
      // We just pushed an entry for the last batch; mark the flag.
      last_batch_first_entry_pushed = true;
    }
  }

  return true;
}

} // namespace sf
