#include <cstdio>
#include <vector>
#include <cstdint>
#include <cstring>
#include "arch/input_spine_buffer.hpp"
#include "arch/min_finder.hpp"
#include "arch/filter_buffer.hpp"
#include "arch/output_queue.hpp"
#include "arch/pe.hpp"

using namespace sf;

// Print non-empty PE outputs for visibility.
static void PrintPEOutputs(const std::vector<int8_t>& outs) {
    for (std::size_t i = 0; i < outs.size(); ++i) {
        if (outs[i] != PE::kNoSpike) {
            std::printf("PE[%zu] spiked at ts=%d\n", i, static_cast<int>(outs[i]));
        }
    }
}

int main() {
    // ------------------------------------------------------------------------
    // 1) Build raw DRAM bytes for the 16 spines and load via LoadSpineFromDRAM
    //    Raw format per Entry: [ts(uint8_t), neuron_id(uint8_t)] repeated.
    // ------------------------------------------------------------------------
    InputSpineBuffer isb;
    constexpr int entries_per_lane = 4;
    for (int lane = 0; lane < kNumSpines; ++lane) {
        std::vector<uint8_t> raw;
        raw.reserve(entries_per_lane * sizeof(Entry));
        for (int i = 0; i < entries_per_lane; ++i) {
            uint8_t ts = static_cast<uint8_t>(1 + lane + 4 * i);
            uint8_t id = static_cast<uint8_t>(lane); // lane maps to filter row id
            raw.push_back(ts);
            raw.push_back(id);
        }
        isb.LoadSpineFromDRAM(lane, raw.data(), raw.size());
    }

    // ------------------------------------------------------------------------
    // 2) Prepare 128 PEs
    // ------------------------------------------------------------------------
    std::vector<PE> pes(FilterBuffer::kNumPEs);

    // ------------------------------------------------------------------------
    // 3) Prepare FilterBuffer with synthetic weights stored in a DRAM-like blob
    //    Each row[j] = (row_id + j) % 5 - 2 creates small signed weights.
    // ------------------------------------------------------------------------
    std::vector<int8_t> dram(FilterBuffer::kTotalRows * FilterBuffer::kRowBytes);
    for (int row = 0; row < FilterBuffer::kTotalRows; ++row) {
        for (int j = 0; j < FilterBuffer::kRowBytes; ++j) {
            int v = (row + j) % 5 - 2;     // in [-2..+2]
            dram[row * FilterBuffer::kRowBytes + j] = static_cast<int8_t>(v);
        }
    }
    FilterBuffer fb;
    fb.LoadFromDRAM(dram.data(), dram.size());

    // ------------------------------------------------------------------------
    // 4) MinFinder arbitrates the 16 spine heads; OutputQueue collects spikes
    // ------------------------------------------------------------------------
    MinFinder   mf(isb);
    OutputQueue oq;

    const int8_t threshold = 3;

    // ------------------------------------------------------------------------
    // 5) Run until all spines are empty:
    //    pick (ts, neuron_id) -> FilterBuffer dispatch row -> 128 PEs process
    //    collect at most one spike into OutputQueue per cycle (by design).
    // ------------------------------------------------------------------------
    while (true) {
        auto pick = mf.PickSmallest();
        if (!pick.has_value()) {
            std::puts("All spines empty. Simulation done.");
            break;
        }

        std::printf("Picked spine %d : (ts=%u, neuron=%u)\n",
                    pick->spine_idx,
                    static_cast<unsigned>(pick->entry.ts),
                    static_cast<unsigned>(pick->entry.neuron_id));

        auto outs = fb.DispatchToPEs(pick->entry, pes, threshold);
        PrintPEOutputs(outs);

        // Collect the single PE spike (if any) into the output queue.
        // Although logic promises at most one, we scan robustly.
        for (std::size_t i = 0; i < outs.size(); ++i) {
            if (outs[i] != PE::kNoSpike) {
                oq.Receive(static_cast<uint8_t>(i), outs[i]); // neuron_id = PE index
                // break; // ignore any additional spikes if they appear
            }
        }
    }

    // ------------------------------------------------------------------------
    // 6) Inputs done -> store the whole OutputQueue back to DRAM as raw bytes.
    //    Buffer layout: [ts0,id0, ts1,id1, ...].
    // ------------------------------------------------------------------------
    const std::size_t bytes_needed = oq.ByteSize();
    std::vector<uint8_t> out_dram(bytes_needed ? bytes_needed : 1); // allocate at least 1 byte
    oq.StoreToDRAM(out_dram.data(), out_dram.size());

    std::printf("OutputQueue stored %zu entries (%zu bytes) back to DRAM.\n",
                oq.Count(), out_dram.size());

    // Optional: print a few entries for sanity.
    std::size_t to_print = oq.Count() < 8 ? oq.Count() : 8;
    for (std::size_t i = 0; i < to_print; ++i) {
        uint8_t ts = out_dram[2*i + 0];
        uint8_t id = out_dram[2*i + 1];
        std::printf("  DRAM[%zu] -> ts=%u, neuron_id=%u\n", i, (unsigned)ts, (unsigned)id);
    }

    return 0;
}
