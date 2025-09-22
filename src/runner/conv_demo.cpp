#include <cstdio>
#include <vector>
#include <cstdint>
#include <cmath>
#include <cstring>
#include "arch/input_spine_buffer.hpp"
#include "arch/min_finder.hpp"
#include "arch/filter_buffer.hpp"
#include "arch/output_queue.hpp"
#include "arch/input_pager.hpp"
#include "arch/pe.hpp"

using namespace sf;

// --- Demo params (you can scale them up to your real case) ---
static constexpr int K    = 3;    // kernel = 3x3 (change to 5 for your case)
static constexpr int Cin  = 8;    // input channels (change to 512)
static constexpr int Cout = 128;  // output channels (change to 512)
static constexpr int H    = 8;    // input height (change to 100)
static constexpr int W    = 8;    // input width (change to 100)
static constexpr int Stride = 1;  // stride (use 1; stride=0 is invalid)
static constexpr int Pad    = 0;  // padding
static constexpr int PEs    = 128;

static_assert(Cout % PEs == 0, "Cout must be a multiple of 128");
static constexpr int CoutTileCount = Cout / PEs;

// ---- Helper: compute output dims ----
static constexpr int OH = (H - K + 2*Pad)/Stride + 1;
static constexpr int OW = (W - K + 2*Pad)/Stride + 1;

// Fake DRAM blobs for inputs (spines) and weights
static std::vector<uint8_t> g_input_dram;   // serialized spines for all (y,x)
static std::vector<int8_t>  g_weight_dram;  // serialized rows for [ky][kx][inC][Cout], in rows of 128

// Map (y,x) -> spine index and get DRAM slice
static inline std::size_t SpineIndex(int y, int x) { return static_cast<std::size_t>(y*W + x); }

static InputPager::RawSlice GetSpineRaw(std::size_t spine_idx) {
    // For demo we store each spine contiguously and equally sized.
    // Let's assume each spine holds S events; here choose S=4 for small demo.
    constexpr std::size_t S = 4;
    constexpr std::size_t bytes_per_spine = S * sizeof(Entry);
    const std::size_t offset = spine_idx * bytes_per_spine;
    return { g_input_dram.data() + offset, bytes_per_spine };
}

// Compute FilterBuffer row_id for (ky,kx,in_c,tile)
static inline int RowID(int ky, int kx, int in_c, int out_tile) {
    return (((ky * K + kx) * Cin) + in_c) * CoutTileCount + out_tile;
}

int main() {
    std::puts("Conv demo starting...");

    // ------------------------ 0) Prefill DRAM ------------------------
    // 0.1) Prefill input DRAM: for each (y,x), build S=4 events [ts,id] with id=in_c.
    constexpr std::size_t S = 4;
    const std::size_t total_spines = static_cast<std::size_t>(H) * W;
    g_input_dram.resize(total_spines * S * sizeof(Entry));
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const std::size_t base = (SpineIndex(y,x) * S * sizeof(Entry));
            for (std::size_t i = 0; i < S; ++i) {
                // Fake timestamps to create interleaving
                uint8_t ts = static_cast<uint8_t>(1 + ((y*W + x) % 5) + 3*i);
                uint8_t id = static_cast<uint8_t>(i % Cin); // in_c in [0..Cin-1] for demo
                g_input_dram[base + 2*i + 0] = ts;
                g_input_dram[base + 2*i + 1] = id;
            }
        }
    }

    // 0.2) Prefill weight DRAM for FilterBuffer:
    // total rows = (K*K*Cin) * CoutTileCount, each row 128 bytes (one weight per PE/out channel in tile).
    const int total_rows = (K*K*Cin) * CoutTileCount;
    g_weight_dram.resize(total_rows * FilterBuffer::kRowBytes);
    for (int ky = 0; ky < K; ++ky) {
        for (int kx = 0; kx < K; ++kx) {
            for (int ic = 0; ic < Cin; ++ic) {
                for (int tile = 0; tile < CoutTileCount; ++tile) {
                    const int row = RowID(ky,kx,ic,tile);
                    for (int j = 0; j < FilterBuffer::kRowBytes; ++j) {
                        // Simple deterministic pattern; real case should come from trained weights
                        int v = ((ky*7 + kx*3 + ic + j + tile*11) % 5) - 2; // [-2..2]
                        g_weight_dram[row*FilterBuffer::kRowBytes + j] = static_cast<int8_t>(v);
                    }
                }
            }
        }
    }

    // --------------------- 1) Load weights into FilterBuffer ---------------------
    FilterBuffer fb;
    fb.LoadFromDRAM(g_weight_dram.data(), g_weight_dram.size());

    // --------------------- 2) Build Pager and ISB/MinFinder ---------------------
    InputSpineBuffer isb;
    InputPager pager(isb, GetSpineRaw);

    // --------------------- 3) Prepare PEs and OutputQueue ------------------------
    std::vector<PE> pes(PEs);
    OutputQueue oq;

    // Threshold selection (demo)
    const int8_t threshold = 3;

    // --------------------- 4) Convolution main loops -----------------------------
    for (int oy = 0; oy < OH; ++oy) {
        for (int ox = 0; ox < OW; ++ox) {
            for (int tile = 0; tile < CoutTileCount; ++tile) {
                // Reset PEs between tiles (optional, depending on your neuron model)
                for (auto& pe : pes) pe = PE();

                // For each kernel position and input channel, process needed spines
                for (int ky = 0; ky < K; ++ky) {
                    for (int kx = 0; kx < K; ++kx) {
                        const int ix = ox*Stride - Pad + kx;
                        const int iy = oy*Stride - Pad + ky;
                        if (ix < 0 || ix >= W || iy < 0 || iy >= H) continue;

                        // Here "required spines" = the single spine at (iy,ix) across channels.
                        // In spike-encoded setting, events embed in_c in Entry.neuron_id.
                        // We must process Cin logically. In demo, each spine carries S events with id in [0..Cin)
                        const std::size_t spine_idx = SpineIndex(iy, ix);

                        // Paginate: (this example has one spine; real case may group many spines per batch)
                        std::size_t done = 0;
                        while (done < 1) {
                            const std::size_t loaded = pager.LoadBatch(done, 1); // load into up to 16 lanes
                            (void)loaded; // 1 spine -> will load into lane 0
                            MinFinder mf(isb);

                            // Consume the batch
                            while (true) {
                                auto pick = mf.PickSmallest();
                                if (!pick.has_value()) break;

                                // pick->entry.neuron_id is in_c for this event
                                const int ic = pick->entry.neuron_id % Cin;

                                // Read proper row = weights[ky][kx][ic][ tile*128 .. ]
                                FilterBuffer::Row row{};
                                const int row_id = RowID(ky,kx,ic,tile);
                                bool ok = fb.ReadRow(row_id, row);
                                if (!ok) {
                                    // fallback to zero-weights if out of bound
                                    for (auto& w : row) w = 0;
                                }

                                // Broadcast to 128 PEs
                                std::vector<int8_t> outs;
                                outs.reserve(PEs);
                                for (int i = 0; i < PEs; ++i) {
                                    const int8_t w = row[i];
                                    const int8_t o = pes[i].Process(static_cast<int8_t>(pick->entry.ts), w, threshold);
                                    outs.push_back(o);
                                }

                                // Collect at most one spike into OutputQueue
                                for (int i = 0; i < PEs; ++i) {
                                    if (outs[i] != PE::kNoSpike) {
                                        oq.Receive(static_cast<uint8_t>(tile*PEs + i), outs[i]); // global out channel id
                                        break;
                                    }
                                }
                            }
                            done += loaded;
                        } // pager while
                    } // kx
                } // ky

            } // tile
        } // ox
    } // oy

    // --------------------- 5) Store output queue back to DRAM --------------------
    std::vector<uint8_t> out_dram(oq.ByteSize() ? oq.ByteSize() : 1);
    oq.StoreToDRAM(out_dram.data(), out_dram.size());
    std::printf("Conv demo done. Output entries: %zu (bytes=%zu)\n", oq.Count(), out_dram.size());

    // Print first few outputs
    for (std::size_t i = 0; i < std::min<std::size_t>(oq.Count(), 12); ++i) {
        std::printf("  out[%zu]: ts=%u, outCh=%u\n", i, out_dram[2*i], out_dram[2*i+1]);
    }
    return 0;
}
