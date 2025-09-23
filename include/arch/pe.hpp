#pragma once
#include <cstdint>

namespace sf {

class PE {
public:
    static constexpr int8_t kNoSpike = -1;

    PE();

    // Process: accumulator -> comparator -> Vmem update -> output generator.
    int8_t Process(int8_t timestamp, int8_t filter, int8_t threshold);

    // Getters for tests
    int8_t  vmem()       const { return V_mem_; }
    bool    spiked()     const { return spiked_; }
    int8_t  reset_vmem() const { return reset_V_mem_; }

    void set_reset_vmem(int8_t v) { reset_V_mem_ = v; }

    // --- New: output neuron id wiring for this PE ---
    void        set_output_neuron_id(std::uint32_t id) { out_neuron_id_ = id; }
    std::uint32_t output_neuron_id() const { return out_neuron_id_; }

private:
    int8_t Accumulator(int8_t V_mem, int8_t filter) const;
    bool   Comparator(int8_t new_possible_V_mem, int8_t threshold) const;
    int8_t VmemUpdate(int8_t new_possible_V_mem, bool spiked, int8_t reset_V_mem) const;
    int8_t OutputGenerator(bool spiked, int8_t timestampRegister) const;

private:
    int8_t       V_mem_;
    bool         spiked_;
    int8_t       reset_V_mem_;
    std::uint32_t out_neuron_id_ = 0xFFFFFFFFu; // invalid by default
};

} // namespace sf
