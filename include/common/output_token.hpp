#pragma once
#include <cstdint>
#include "common/entry.hpp"

namespace sf {

// A single output produced by a PE at some timestep and belonging to a certain (batch, spine).
struct OutputToken {
    int      pe_id    = 0;   // Physical PE index [0..kNumPEs-1]
    int      t        = 0;   // Timestep index (optional; set if useful)
    Entry    value{};        // Payload (reuse your existing Entry type)
    bool     is_eos  = false;// Optional: marks end-of-spine if you choose to use EOS tokens
};

} // namespace sf
