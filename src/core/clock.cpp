// core/clock.cpp
#include "core/clock.hpp"

namespace sf {

ClockCore::ClockCore(std::size_t outq_capacity)
  : out_q_(outq_capacity)
  , min_finder_(in_buf_)  // S4 consumes from PSB
{
  // S0
  out_q_.RegisterCore(this);
  // S1
  ts_picker_.RegisterCore(this);
  // S2
  pe_array_.RegisterCore(this);
  // S3
  iwp_.RegisterCore(this);
  // S4
  min_finder_.RegisterCore(this);
  // S5
  in_buf_.RegisterCore(this);
}

bool ClockCore::run() {
  const bool s0 = out_q_.run();               // Stage 0: receive & assemble only
  const bool s1 = ts_picker_.run();           // Stage 1
  const bool s2 = pe_array_.run();            // Stage 2
  const bool s3 = iwp_.run();                 // Stage 3
  const bool s4 = min_finder_.run();          // Stage 4
  const bool s5 = in_buf_.run();              // Stage 5 (DRAM -> PSB)
  return s0 || s1 || s2 || s3 || s4 || s5;
}

} // namespace sf
