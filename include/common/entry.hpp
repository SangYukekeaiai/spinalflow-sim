#pragma once
#include <cstdint>

/* All comments are in English.
 * Common Entry struct shared by buffers, min_finder, and output queue.
 */

namespace sf {

// One spike entry: timestamp + neuron_id
struct Entry {
    uint8_t  ts;         // timestamp when spike arrives
    uint32_t neuron_id;  // which neuron fired (32-bit to cover large tensors)
};

} // namespace sf
