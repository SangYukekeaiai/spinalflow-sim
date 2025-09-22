#include "arch/output_serializer.hpp"

namespace sf {

std::size_t OutputSerializer::Drain(const OutputHarvest& H) {
    switch (mode_) {
        case Mode::RoundRobinPE:
            return drainRoundRobin(H);
        default:
            return drainRoundRobin(H);
    }
}

std::size_t OutputSerializer::drainRoundRobin(const OutputHarvest& H) {
    // Iterate only over PEs that actually produced (IndexFIFO order).
    std::size_t enq = 0;
    H.ForEach([&](int pe, const OutputToken& tok){
        if (q_.full()) return; // stop early if queue is full
        if (q_.push(tok)) ++enq;
    });
    return enq;
}


} // namespace sf
