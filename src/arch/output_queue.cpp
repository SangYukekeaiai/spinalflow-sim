#include "arch/output_queue.hpp"

namespace sf {

void OutputQueue::Receive(uint8_t pe_idx, int8_t ts) {
    Entry e;
    e.ts = static_cast<uint8_t>(ts); // PE outputs int8_t; store as uint8_t
    e.neuron_id = pe_idx;           // use PE index as output neuron id
    q_.push_back(e);
}

void OutputQueue::StoreToDRAM(uint8_t* base, std::size_t length) const {
    if (!base && !q_.empty()) {
        throw std::invalid_argument("OutputQueue::StoreToDRAM: null base with non-empty queue");
    }
    const std::size_t need = ByteSize();
    if (length < need) {
        throw std::length_error("OutputQueue::StoreToDRAM: destination buffer too small");
    }
    // Serialize as [ts, id] pairs
    for (std::size_t i = 0; i < q_.size(); ++i) {
        base[2*i + 0] = q_[i].ts;
        base[2*i + 1] = q_[i].neuron_id;
    }
}

} // namespace sf
