#include "arch/pe.hpp"
#include <limits>

namespace sf {

PE::PE() : V_mem_(0), spiked_(false), reset_V_mem_(0), out_neuron_id_(0xFFFFFFFFu) {}

int8_t PE::Accumulator(int8_t V_mem, int8_t filter) const {
    int16_t sum = static_cast<int16_t>(V_mem) + static_cast<int16_t>(filter);
    if (sum > std::numeric_limits<int8_t>::max()) sum = std::numeric_limits<int8_t>::max();
    if (sum < std::numeric_limits<int8_t>::min()) sum = std::numeric_limits<int8_t>::min();
    return static_cast<int8_t>(sum);
}

bool PE::Comparator(int8_t new_possible_V_mem, int8_t threshold) const {
    return new_possible_V_mem >= threshold;
}

int8_t PE::VmemUpdate(int8_t new_possible_V_mem, bool spiked, int8_t reset_V_mem) const {
    return spiked ? reset_V_mem : new_possible_V_mem;
}

int8_t PE::OutputGenerator(bool spiked, int8_t timestampRegister) const {
    return spiked ? timestampRegister : kNoSpike;
}

int8_t PE::Process(int8_t timestamp, int8_t filter, int8_t threshold) {
    const int8_t new_possible_V = Accumulator(V_mem_, filter);
    spiked_ = Comparator(new_possible_V, threshold);
    V_mem_  = VmemUpdate(new_possible_V, spiked_, reset_V_mem_);
    return OutputGenerator(spiked_, timestamp);
}

} // namespace sf
