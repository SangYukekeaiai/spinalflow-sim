#pragma once
#include <cstdint>

/* All comments are in English.
 * PE model as per user's spec:
 * Members:
 *   - V_mem        : int8_t
 *   - spiked       : bool
 *   - reset_V_mem  : int8_t
 * Public:
 *   - Process(timestamp, filter, threshold) -> int8_t (encoded output)
 * Private helpers (invoked by Process):
 *   2) Accumulator
 *   3) Comparator
 *   4) V mem update
 *   5) Output generator
 */

namespace sf {

class PE {
public:
    // Special value to indicate "no spike" in Process() output.
    static constexpr int8_t kNoSpike = -1;

    // Constructor: V_mem(0), spiked(false), reset_V_mem(0)
    PE();

    // Process step: runs 2) Accumulator -> 3) Comparator -> 4) Vmem update -> 5) Output generator.
    // Inputs:
    //   - timestamp : current tick (int8_t by user's spec)
    //   - filter    : weight for this PE at this tick (int8_t)
    //   - threshold : firing threshold (int8_t)
    // Returns:
    //   - output(int8_t): timestamp if spiked this call, or kNoSpike if not.
    int8_t Process(int8_t timestamp, int8_t filter, int8_t threshold);

    // Getters for tests
    int8_t  vmem()  const { return V_mem_; }
    bool    spiked() const { return spiked_; }
    int8_t  reset_vmem() const { return reset_V_mem_; }

    // Allow configuring reset value if needed.
    void set_reset_vmem(int8_t v) { reset_V_mem_ = v; }

private:
    // 2) Accumulator: new_possible_V_mem = V_mem + filter (with saturating add)
    int8_t Accumulator(int8_t V_mem, int8_t filter) const;

    // 3) Comparator: spiked? = (new_possible_V_mem >= Threshold)
    bool Comparator(int8_t new_possible_V_mem, int8_t threshold) const;

    // 4) V mem update: if(spiked) => reset_V_mem; else => new_possible_V_mem
    int8_t VmemUpdate(int8_t new_possible_V_mem, bool spiked, int8_t reset_V_mem) const;

    // 5) Output generator: if(spiked) => timestampRegister; else => kNoSpike
    int8_t OutputGenerator(bool spiked, int8_t timestampRegister) const;

private:
    int8_t V_mem_;
    bool   spiked_;
    int8_t reset_V_mem_;
};

} // namespace sf
