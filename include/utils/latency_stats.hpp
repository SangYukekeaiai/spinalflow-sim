#pragma once
#include <array>
#include <cstdint>

namespace sf {
namespace util {

/* Lightweight queue latency accumulators and stage counters.
 * Separated from arch to keep stats tooling out of the hardware path.
 */

struct QueueLatencyStats {
    std::uint64_t count = 0;  // number of completed items measured
    std::uint64_t total = 0;  // sum of wait cycles over all completed items
    std::uint64_t max   = 0;  // maximum single-item wait cycles observed
};

struct StageCounters {
    std::uint64_t steps       = 0;                 // total cycles (Step calls)
    std::uint64_t idle_cycles = 0;                 // cycles when no stage did useful work
    std::array<std::uint64_t, 7> stage_hits{};     // per-stage useful-work hits (stages 0..6)
};

struct LatencyStats {
    StageCounters     stages;        // per-cycle & per-stage counters
    QueueLatencyStats fifo_wait;     // wait inside per-batch IntermediateFIFO (enqueue->pick+pop)
    QueueLatencyStats output_queue;  // wait inside OutputQueue (enqueue->S0 pop to sink)
};

// Add one observation to a QueueLatencyStats.
inline void AccumulateQueueLatency(QueueLatencyStats& dst, std::uint64_t latency) {
    dst.count += 1;
    dst.total += latency;
    if (latency > dst.max) dst.max = latency;
}

} // namespace util
} // namespace sf
