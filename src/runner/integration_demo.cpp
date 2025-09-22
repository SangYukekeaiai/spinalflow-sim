#include <cstdio>
#include <vector>
#include "arch/input_spine_buffer.hpp"
#include "arch/min_finder.hpp"
#include "arch/filter_buffer.hpp"
#include "arch/pe.hpp"

using namespace sf;

static void PrintPEOutputs(const std::vector<int8_t>& outs) {
    for (std::size_t i = 0; i < outs.size(); ++i) {
        if (outs[i] != PE::kNoSpike) {
            std::printf("PE[%zu] spiked at ts=%d\n", i, static_cast<int>(outs[i]));
        }
    }
}

int main() {
    // 1) Prepare InputSpineBuffer with staggered timestamps
    InputSpineBuffer isb;
    for (int lane = 0; lane < kNumSpines; ++lane) {
        Entry tmp[4];
        for (int i = 0; i < 4; ++i) {
            tmp[i].ts = static_cast<uint8_t>(1 + lane + 4 * i);
            tmp[i].neuron_id = static_cast<uint8_t>(lane); // map lane to row id in FilterBuffer
        }
        isb.LoadSpine(lane, tmp, 4);
    }

    // 2) Prepare 128 PEs
    std::vector<PE> pes(FilterBuffer::kNumPEs);

    // 3) Prepare FilterBuffer and load synthetic weights from DRAM-like memory.
    //    We'll create kTotalRows rows; each row[j] = (row_id + j) % 5 - 2 in [-2..2].
    std::vector<int8_t> dram(FilterBuffer::kTotalRows * FilterBuffer::kRowBytes);
    for (int row = 0; row < FilterBuffer::kTotalRows; ++row) {
        for (int j = 0; j < FilterBuffer::kRowBytes; ++j) {
            int v = (row + j) % 5 - 2; // simple pattern
            dram[row * FilterBuffer::kRowBytes + j] = static_cast<int8_t>(v);
        }
    }

    FilterBuffer fb;
    fb.LoadFromDRAM(dram.data(), dram.size());

    // 4) MinFinder arbitrates spine heads
    MinFinder mf(isb);

    const int8_t threshold = 3;

    // 5) Run until all spines are empty
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

        // FilterBuffer fetches row (by neuron_id) and broadcasts weights to the PEs
        auto outs = fb.DispatchToPEs(pick->entry, pes, threshold);
        PrintPEOutputs(outs);
    }

    return 0;
}
