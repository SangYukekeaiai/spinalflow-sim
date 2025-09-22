// tests/test_pipeline_integration.cpp
// Integration test for: InputSpineBuffer (double-buffer) + MinFinderBatch (intra-batch)
// + four IntermediateFIFO(256B) + GlobalMerger (4-way merge).
// All comments in English as requested.

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "common/entry.hpp"                 // assumes: struct Entry { uint8_t ts; uint8_t neuron_id; };
#include "arch/input_spine_buffer.hpp"      // revised double-buffer version
#include "arch/min_finder_batch.hpp"        // DrainBatchInto(...)
#include "arch/intermediate_fifo.hpp"       // 256B = 128 entries
#include "arch/global_merger.hpp"           // 4-way PickAndPop(...)
using namespace sf;
// ---------------------------- Layer configuration ----------------------------
enum class LayerType { kConv, kFC };

struct LayerConfig {
    LayerType type;
    int in_channels;    // number of input channels
    int kernel_h;       // kernel height (for conv)
    int kernel_w;       // kernel width (for conv)
};

// ---------------------------- DRAM model (byte array) ------------------------
struct SpineSpan {
    // Byte span inside dram for one logical spine.
    std::size_t offset_bytes = 0;
    std::size_t size_bytes   = 0;
};

struct DRAM {
    // Base is conceptually 0; we keep a single contiguous array of bytes.
    std::vector<uint8_t> bytes;
    std::vector<SpineSpan> spines; // spines[i] is the byte span for logical spine i
};

// Utility to append one Entry to dram bytes as raw (ts, neuron_id).
static inline void push_entry_bytes(std::vector<uint8_t>& b, const Entry& e) {
    b.push_back(e.ts);
    b.push_back(e.neuron_id);
}

// ----------------------------- Test data generator ---------------------------
// Build DRAM with "spine-format": spine0 holds position(0,0) across channels [0..127],
// spine1 holds channels [128..255], etc. Each spine stores a sorted-by-ts list of Entry.
// NOTE: We deliberately create overlapping ts ranges across spines to exercise the 4-way merge.
DRAM BuildDramSpineFormat(const LayerConfig& cfg) {
    DRAM dram;
    dram.bytes.reserve(1 << 20); // reserve some space

    const int group = 128; // 128 channels per logical spine (as per requirement)
    const int num_spines = (cfg.in_channels + group - 1) / group;

    dram.spines.resize(num_spines);

    // Generate entries for position(0,0) across channels, grouped by 128 per spine.
    // For each channel c in this spine, we create one Entry:
    //   ts: crafted to create overlaps across spines while preserving per-spine sorted order.
    //   neuron_id: channel index (fits in uint8_t for <=255; if >255, wrap for demo).
    //
    // To ensure per-spine sorted(ts), we generate (ts = c % 64), then sort by ts then neuron_id.
    // This creates cross-spine overlaps (0..63) so the global merger is exercised.
    std::size_t cursor = 0;
    std::vector<Entry> tmp; tmp.reserve(group);

    for (int s = 0; s < num_spines; ++s) {
        tmp.clear();
        const int c_begin = s * group;
        const int c_end   = std::min(c_begin + group, cfg.in_channels);

        for (int c = c_begin; c < c_end; ++c) {
            Entry e;
            e.ts        = static_cast<uint8_t>(c % 64);      // intentionally overlapping ts
            e.neuron_id = static_cast<uint8_t>(c & 0xFF);    // pack channel id in 8-bit
            tmp.push_back(e);
        }

        // Sort per spine by (ts, neuron_id) so spine data is monotonically non-decreasing in ts.
        std::sort(tmp.begin(), tmp.end(), [](const Entry& a, const Entry& b){
            if (a.ts != b.ts) return a.ts < b.ts;
            return a.neuron_id < b.neuron_id;
        });

        // Record span and append to dram bytes as raw pairs.
        dram.spines[s].offset_bytes = cursor;
        dram.spines[s].size_bytes   = tmp.size() * sizeof(Entry);
        for (const auto& e : tmp) {
            push_entry_bytes(dram.bytes, e);
        }
        cursor += dram.spines[s].size_bytes;
    }

    return dram;
}

// ----------------------------- Batch orchestration --------------------------
// Map logical spines [0..N) into batches of up to 16 physical lanes each.
struct BatchPlan {
    // Each batch contains indices of logical spines mapped to 16 physical lanes at most.
    std::vector<std::vector<int>> batches; // batches[b][i] = logical spine idx for lane i (size<=16)
};

// Simple planner: pack spines in order into groups of 16, last batch may be <16.
BatchPlan MakeBatches(int num_spines) {
    BatchPlan plan;
    const int lanes = sf::kNumSpinesPhysical; // 16
    for (int s = 0; s < num_spines; ) {
        std::vector<int> batch;
        for (int i = 0; i < lanes && s < num_spines; ++i, ++s) {
            batch.push_back(s);
        }
        plan.batches.push_back(std::move(batch));
    }
    return plan;
}

// Load one batch's logical spines into SHADOW of physical lanes [0..size-1], then swap to ACTIVE.
void LoadBatchIntoShadowAndSwap(const DRAM& dram, const std::vector<int>& logical_spines,
                                sf::InputSpineBuffer& inbuf) {
    // Fill SHADOW
    for (int lane = 0; lane < static_cast<int>(logical_spines.size()); ++lane) {
        int sidx = logical_spines[lane];
        const auto& span = dram.spines[sidx];
        const uint8_t* raw = span.size_bytes ? (&dram.bytes[span.offset_bytes]) : nullptr;
        inbuf.LoadSpineShadowFromDRAM(lane, raw, span.size_bytes);
    }
    // Clear unused lanes' shadow to empty
    for (int lane = static_cast<int>(logical_spines.size()); lane < sf::kNumSpinesPhysical; ++lane) {
        inbuf.LoadSpineShadowFromDRAM(lane, nullptr, 0);
    }
    // Swap all lanes that have data
    for (int lane = 0; lane < static_cast<int>(logical_spines.size()); ++lane) {
        bool ok = inbuf.SwapToShadow(lane);
        (void)ok; // if a spine is empty, swap may be false; it's fine for edge cases.
    }
}

// ------------------------------ Simple assertions ---------------------------
static void AssertNonDecreasingTs(const std::vector<Entry>& v) {
    for (std::size_t i = 1; i < v.size(); ++i) {
        if (v[i].ts < v[i-1].ts) {
            std::cerr << "Global order violated at index " << i
                      << ": prev.ts=" << int(v[i-1].ts)
                      << ", curr.ts=" << int(v[i].ts) << "\n";
            std::abort();
        }
    }
}

static std::size_t TotalEntries(const DRAM& dram) {
    std::size_t sum = 0;
    for (const auto& sp : dram.spines) sum += (sp.size_bytes / sizeof(Entry));
    return sum;
}

// ------------------------------------ main ----------------------------------
int main() {
    // ---------------- Params you may tweak for quick experiments --------------
    LayerConfig cfg;
    cfg.type        = LayerType::kConv; // not used in this test’s data generation, but plumbed in
    cfg.in_channels = 256;              // try 256, 320, 49, etc.
    cfg.kernel_h    = 7;
    cfg.kernel_w    = 7;

    // Build DRAM in the required "spine-format". Base address conceptually 0.
    DRAM dram = BuildDramSpineFormat(cfg);
    const int num_spines = static_cast<int>(dram.spines.size());
    const std::size_t expected_total = TotalEntries(dram);

    std::cout << "[Info] Built DRAM with " << num_spines << " logical spines, total entries = "
              << expected_total << "\n";

    // Make batch plan: pack logical spines into groups of 16 (Step1..Step4 in your diagram).
    BatchPlan plan = MakeBatches(num_spines);
    assert(plan.batches.size() >= 1 && plan.batches.size() <= 4 && "Expect up to 4 batches for 49");

    // Create the building blocks.
    sf::InputSpineBuffer inbuf;          // 16 physical lanes, double-buffered
    sf::MinFinderBatch   batchFinder(inbuf);
    sf::IntermediateFIFO fifo0, fifo1, fifo2, fifo3;
    sf::GlobalMerger     merger;

    // Drain each batch into its FIFO (Step1..Step4).
    // We map batches in order to FIFO0..FIFO3 respectively.
    sf::IntermediateFIFO* fifos[4] = { &fifo0, &fifo1, &fifo2, &fifo3 };
    for (std::size_t b = 0; b < plan.batches.size(); ++b) {
        auto* target = fifos[b];
        LoadBatchIntoShadowAndSwap(dram, plan.batches[b], inbuf);
        std::size_t pushed = batchFinder.DrainBatchInto(*target);
        std::cout << "[Info] Batch " << b << " -> FIFO" << b
                  << " pushed entries = " << pushed << "\n";
    }

    // Global 4-way merge (Step5).
    std::vector<Entry> out_stream;
    out_stream.reserve(expected_total);
    std::array<sf::IntermediateFIFO*, 4> fs = { &fifo0, &fifo1, &fifo2, &fifo3 };
    while (true) {
        auto e = merger.PickAndPop(fs);
        if (!e) break;
        out_stream.push_back(*e);
    }

    // ------------------------------ Validation --------------------------------
    if (out_stream.size() != expected_total) {
        std::cerr << "Entry count mismatch: got " << out_stream.size()
                  << ", expected " << expected_total << "\n";
        return 1;
    }
    AssertNonDecreasingTs(out_stream);

    std::cout << "[PASS] Integration test succeeded. Global non-decreasing order verified. "
              << "Total entries = " << out_stream.size() << "\n";
    return 0;
}
