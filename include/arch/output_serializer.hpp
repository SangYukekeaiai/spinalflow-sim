#pragma once
#include <vector>
#include <algorithm>
#include "arch/output_harvest.hpp"
#include "arch/output_queue.hpp"

namespace sf {

// Deterministic serialization policies for draining the per-step harvest into the OutputQueue.
class OutputSerializer {
public:
    enum class Mode {
        RoundRobinPE,     // use the capture order (IndexFIFO order); typically per-PE ascending
        BySpineThenPE     // stable order by (spine_id, pe_id)
    };

    explicit OutputSerializer(OutputQueue& q, Mode m = Mode::RoundRobinPE)
      : q_(q), mode_(m) {}

    // Drain one step's harvest into the global queue using the selected ordering.
    // Returns the number of enqueued tokens (could be less if the queue hits capacity).
    std::size_t Drain(const OutputHarvest& H);

private:
    OutputQueue& q_;
    Mode mode_;

    std::size_t drainRoundRobin(const OutputHarvest& H);
    std::size_t drainBySpineThenPE(const OutputHarvest& H);
};

} // namespace sf
