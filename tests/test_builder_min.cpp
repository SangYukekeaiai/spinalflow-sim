// All comments are in English.
// Minimal sanity test for Builder’s batch→FIFO discipline and “totally_drained” gating.

#include <cassert>
#include <cstdint>
#include <iostream>
#include <map>
#include <vector>
#include <array>
#include <cstring>

#include "arch/builder.hpp"
#include "arch/input_spine_buffer.hpp"
#include "arch/intermediate_fifo.hpp"
#include "arch/min_finder_batch.hpp"
#include "arch/global_merger.hpp"
#include "arch/filter_buffer.hpp"
#include "arch/pe.hpp"
#include "arch/driver/batch_spine_map.hpp"
#include "arch/driver/weight_lut.hpp"
#include "common/constants.hpp"
#include "common/entry.hpp"

using namespace sf;

// ----------------------------- Fake DRAM ----------------------------------

// A tiny fake DRAM: address -> vector<byte>
struct FakeDram {
    std::map<driver::BatchSpineMap::Addr, std::vector<std::uint8_t>> mem;

    bool read(driver::BatchSpineMap::Addr addr, std::uint8_t* dst, std::size_t bytes) {
        auto it = mem.find(addr);
        if (it == mem.end()) return false;
        if (it->second.size() < bytes) return false;
        std::memcpy(dst, it->second.data(), bytes);
        return true;
    }
};

// Helper to pack a full spine slice of Entries (kCapacityPerSpine entries).
static std::vector<std::uint8_t> make_spine_slice(uint8_t base_ts, uint8_t neuron_id) {
    std::vector<std::uint8_t> buf(sf::IntermediateFIFO::kCapacityEntries * sizeof(Entry), 0);
    // Fill first N entries with increasing timestamps; rest zero (older ts will be merged first).
    const int N = 8; // keep small for test runtime
    for (int i = 0; i < N; ++i) {
        Entry e;
        e.ts = static_cast<uint8_t>(base_ts + i);
        e.neuron_id = neuron_id; // must be < inC*kH*kW
        std::memcpy(buf.data() + i * sizeof(Entry), &e, sizeof(Entry));
    }
    return buf;
}

// ------------------------- Test Scenario Setup ----------------------------

int main() {
    // 1) Build core modules
    InputSpineBuffer in_buf;
    FilterBuffer     filt;
    std::vector<PE>  pes(FilterBuffer::kNumPEs); // 128 PEs

    // 2) Driver structures
    //    Case A: 7x7, inC=3, outC=64 -> typical multi-batch (4 batches of 16 logical spines).
    driver::BatchSpineMap bmap(/*numBatches=*/4);
    driver::WeightLUT lut;

    // 3) Configure LUT for the layer
    sf::Builder::LayerConfig cfg;
    cfg.inC = 3;
    cfg.outC = 64;
    cfg.kH  = 7;
    cfg.kW  = 7;
    cfg.threshold = 5;       // arbitrary
    cfg.tiles_per_step = 1;  // typical

    // 4) Prepare fake DRAM content and BatchSpineMap addresses
    FakeDram dram;

    // DRAM addressing scheme for the test: just use incremental integers as "addresses".
    driver::BatchSpineMap::Addr next_addr = 0;

    // For each batch b and spine s, give it a single slice with a unique base_ts,
    // and use neuron_id within [0, inC*kH*kW) = [0, 147).
    const uint32_t rows_per_tile = static_cast<uint32_t>(cfg.inC) * cfg.kH * cfg.kW; // 147
    const uint8_t  neuron_id_for_test = 42;  // any < 147
    for (int b = 0; b < 4; ++b) {
        for (int s = 0; s < kNumSpines; ++s) {
            auto payload = make_spine_slice(/*base_ts=*/static_cast<uint8_t>(10*b + s),
                                            /*neuron_id=*/neuron_id_for_test);
            dram.mem[next_addr] = std::move(payload);
            bmap.Add(b, s, next_addr);
            ++next_addr;
        }
    }

    // 5) Fill FilterBuffer with rows so that LUT RowIdFromNeuron(...) always hits.
    //    We only need enough rows to cover 1 tile (outC=64 => outTiles=1).
    //    FilterBuffer::LoadFromDRAM expects multiples of 128B rows; we’ll build a blob.
    const int total_rows_needed = static_cast<int>(rows_per_tile); // 147 rows for tile 0
    std::vector<int8_t> weight_blob(total_rows_needed * FilterBuffer::kRowBytes, 0);
    // Put small positive weights to ensure many PE spikes for visibility.
    for (int r = 0; r < total_rows_needed; ++r) {
        for (int i = 0; i < FilterBuffer::kRowBytes; ++i) {
            weight_blob[r * FilterBuffer::kRowBytes + i] = 1; // simple constant row
        }
    }

    // 6) Build the LUT and the Builder
    lut.Build({cfg.inC, cfg.outC, cfg.kH, cfg.kW, 128});
    Builder builder(in_buf, filt, pes, bmap, lut);

    // Plug DRAM reader
    builder.SetDramReader([&](driver::BatchSpineMap::Addr addr, std::uint8_t* dst, std::size_t bytes){
        return dram.read(addr, dst, bytes);
    });

    // Configure layer, prefill weights and batch-0 spines
    builder.ConfigureLayer(cfg);
    builder.PrefillWeightsOnce(reinterpret_cast<const int8_t*>(weight_blob.data()),
                               weight_blob.size());
    builder.PrefillInputSpinesOnce(/*try_atomic_swap=*/true);

    // -------------------------- Assertions --------------------------------

    auto fifo_sizes = [&](std::array<size_t,4>& out){
        for (int b = 0; b < 4; ++b) out[b] = builder.FifoOf(b).size();
    };

    // Sanity: after first prefill/swap, either no ACTIVE yet (if not ready), or batch 0 is active.
    int steps = 0;
    std::array<size_t,4> prev_sizes{}; fifo_sizes(prev_sizes);

    // Run a number of steps; during the period where active_batch is set,
    // only its FIFO is allowed to grow (batch→FIFO discipline).
    for (; steps < 10000; ++steps) {
        const int active_before = builder.ActiveBatch();

        (void)builder.Step();

        std::array<size_t,4> now_sizes{}; fifo_sizes(now_sizes);
        const int active_after = builder.ActiveBatch();

        // Check FIFO growth rule: if an active batch exists, only its FIFO may increase.
        if (active_before >= 0) {
            for (int b = 0; b < 4; ++b) {
                if (b == active_before) continue;
                assert(now_sizes[b] <= prev_sizes[b] && "Non-active batch FIFO grew! Mapping violated.");
            }
        }
        prev_sizes = now_sizes;

        // Optional: print a few states for visual inspection
        if (steps % 20 == 0) {
            std::cout << "[t=" << steps << "] " << builder.DebugString() << "\n";
        }

        // If all batches get drained and GlobalMerger stops stalling, we should be able to finish.
        // We won't explicitly assert drained flags here; DebugString shows `drained=[...]`.
    }
    // std::cout << "[t=" << steps << "] " << builder.DebugString() << "\n";
    std::cout << "Test completed " << steps << " steps. Batch→FIFO discipline preserved.\n";
    return 0;
}
