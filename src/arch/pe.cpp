#include "arch/pe.hpp"
#include <limits>

/* All comments are in English */

namespace sf {

PE::PE() : V_mem_(0), spiked_(false), reset_V_mem_(0) {}

int8_t PE::Accumulator(int8_t V_mem, int8_t filter) const {
    // Use int16_t to avoid signed overflow UB, then clamp to int8_t (saturating add).
    int16_t sum = static_cast<int16_t>(V_mem) + static_cast<int16_t>(filter);
    if (sum > std::numeric_limits<int8_t>::max()) sum = std::numeric_limits<int8_t>::max();
    if (sum < std::numeric_limits<int8_t>::min()) sum = std::numeric_limits<int8_t>::min();
    return static_cast<int8_t>(sum);
}

bool PE::Comparator(int8_t new_possible_V_mem, int8_t threshold) const {
    // Fire when potential reaches or exceeds threshold.
    return new_possible_V_mem >= threshold;
}

int8_t PE::VmemUpdate(int8_t new_possible_V_mem, bool spiked, int8_t reset_V_mem) const {
    // If spiked, reset to reset_V_mem; otherwise keep the new value.
    return spiked ? reset_V_mem : new_possible_V_mem;
}

int8_t PE::OutputGenerator(bool spiked, int8_t timestampRegister) const {
    // When spiked, output timestamp; otherwise output kNoSpike.
    return spiked ? timestampRegister : kNoSpike;
}

int8_t PE::Process(int8_t timestamp, int8_t filter, int8_t threshold) {
    // 2) Accumulator
    const int8_t new_possible_V = Accumulator(V_mem_, filter);

    // 3) Comparator
    spiked_ = Comparator(new_possible_V, threshold);

    // 4) V mem update
    V_mem_ = VmemUpdate(new_possible_V, spiked_, reset_V_mem_);

    // 5) Output generator
    return OutputGenerator(spiked_, timestamp);
}

} // namespace sf
