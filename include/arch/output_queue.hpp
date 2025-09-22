#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include <stdexcept>
#include "common/entry.hpp"

namespace sf {

/**
 * OutputQueue
 * - Collects spikes from 128 PEs over time.
 * - Each collected output is stored as Entry{ts, neuron_id}, where neuron_id is PE index.
 * - Once the whole input processing finishes, the queue can be stored back to DRAM
 *   as raw bytes [ts, id] repeated.
 */
class OutputQueue {
public:
    // Remove all stored outputs.
    void Reset() { q_.clear(); }

    // Receive one output from a PE. Caller guarantees at most one spike per cycle overall.
    // 'pe_idx' becomes the output neuron_id; 'ts' is the timestamp carried by the PE output.
    void Receive(uint8_t pe_idx, int8_t ts);

    // Number of stored entries and byte size when serialized to [ts,id] bytes.
    std::size_t Count() const { return q_.size(); }
    std::size_t ByteSize() const { return q_.size() * sizeof(Entry); }

    // Store the whole queue back to DRAM in raw byte format [ts, id, ts, id, ...].
    // 'length' must be >= ByteSize(). Throws if not enough.
    void StoreToDRAM(uint8_t* base, std::size_t length) const;

    // Optional: expose const view for tests.
    const std::vector<Entry>& Data() const { return q_; }

private:
    std::vector<Entry> q_;
};

} // namespace sf
