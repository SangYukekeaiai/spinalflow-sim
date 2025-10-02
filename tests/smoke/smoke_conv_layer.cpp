// tests/smoke/smoke_conv_runner.cpp
// All comments in English.

#include <cassert>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cstring>

#include "common/entry.hpp"
#include "core/clock.hpp"

#include "arch/dram/dram_common.hpp"
#include "arch/dram/dram_format.hpp"
#include "arch/dram/fixed_stride_format.hpp"
#include "arch/dram/stream_reader.hpp"
#include "arch/dram/stream_writer.hpp"     // assumed available
#include "arch/dram/layer_directory.hpp"
#include "arch/dram/input_spine_fetcher.hpp"
#include "arch/dram/output_spine_writer.hpp"
#include "arch/dram/weight_loader.hpp"
#include "arch/dram/conv_shape.hpp"

#include "runner/conv_runner.hpp"

using namespace sf;
using namespace sf::dram;

static void check(bool cond, const char* msg) {
  if (!cond) {
    std::fprintf(stderr, "SMOKE (runner) FAILED: %s\n", msg);
    std::abort();
  }
}

int main() {
  std::printf("[runner-smoke] start\n");

  // ---- 0) Backing DRAM (RW) + read-only view ----
  std::vector<std::uint8_t> dram_bytes;
  dram_bytes.reserve(1 << 20);
  DramImage img;
  img.bind(dram_bytes.data(), dram_bytes.size());

  // ---- 1) Directory with 2 layers (L=0, L=1) ----
  LayerDirectory dir;
  dir.reset(/*num_layers=*/2);

  // ---- 2) Formats ----
  FixedStrideFormat spine_fmt (sizeof(SegmentHeader), sizeof(Entry), 128);
  FixedStrideFormat weight_fmt(sizeof(SegmentHeader), /*entry_bytes=*/1, 128);

  // ---- 3) Seed minimal inputs for L=0 (so PSB can read something) ----
  // Create one small spine stream for (spine_id=0): 1 segment with 3 entries.
  {
    StreamWriter wr(spine_fmt, dram_bytes);

    SegmentHeader h{};
    h.version          = 1;
    h.kind             = SEG_SPINE;
    h.layer_id         = 0;
    h.logical_spine_id = 0;         // spine 0
    h.size             = 3;         // 3 entries payload
    h.seg_id           = 0;
    h.seg_count        = 0;
    h.eol              = 1;         // last segment
    h.aux0             = 0;
    h.aux1             = 0;
    h.reserved         = 0;

    Entry ents[3]{};
    ents[0].neuron_id = 0; ents[0].ts = 10;
    ents[1].neuron_id = 1; ents[1].ts = 11;
    ents[2].neuron_id = 2; ents[2].ts = 12;

    const std::uint64_t ibeg = dram_bytes.size();
    wr.append(h, reinterpret_cast<const std::uint8_t*>(ents), sizeof(ents));
    const std::uint64_t iend = dram_bytes.size();

    dir.set_input_range(0, Range{ibeg, iend});
  }

  // ---- 4) Seed minimal weights for L=0 (one row for oc_group=0) ----
  {
    StreamWriter wr(weight_fmt, dram_bytes);

    SegmentHeader h{};
    h.version          = 1;
    h.kind             = SEG_WEIGHT;
    h.layer_id         = 0;
    h.logical_spine_id = 0;  // kk_idx = ky*KW + kx = 0 for KH=KW=1
    h.size             = 128;
    h.seg_id           = 0;
    h.seg_count        = 0;
    h.eol              = 1;
    h.aux0             = 0;  // inC
    h.aux1             = 0;  // oc_group
    h.reserved         = 0;

    std::uint8_t row[128];
    for (int i = 0; i < 128; ++i) row[i] = static_cast<std::uint8_t>(i);

    const std::uint64_t wbeg = dram_bytes.size();
    wr.append(h, row, sizeof(row));
    const std::uint64_t wend = dram_bytes.size();

    dir.set_weights_range(0, Range{wbeg, wend});
  }

  // Rebind RO view after appends
  img.bind(dram_bytes.data(), dram_bytes.size());

  // ---- 5) Build core and prefill OutputQueue to exercise the writer ----
  ClockCore core;
  // Prefill some entries that will be drained to the sink (writer) during run().
  // We map spine = neuron_id % (H*W) in the SpineIdFn below.
  {
    auto& oq = core.output_queue();
    for (int i = 0; i < 10; ++i) {
      Entry e{};
      e.neuron_id = static_cast<uint16_t>(i); // used by SpineIdFn
      e.ts        = static_cast<uint8_t>(100 + i);
      (void)oq.push_entry(e);
    }
  }

  // ---- 6) Prepare ConvRunner spec (KH=KW=IC=1 for simplicity) ----
  ConvLayerSpec spec;
  spec.L        = 0;
  spec.H        = 1;  // we only seeded spine 0 in inputs
  spec.W        = 2;  // we will round-robin outputs into 2 spines via SpineIdFn
  spec.OC_tiles = 1;  // only oc_group=0
  spec.shape    = ConvShape{1,1,16};

  // ---- 7) Build runner (note: BOTH RO image and RW bytes, plus mutable dir) ----
  ConvRunnerOptions opt;
  
  opt.idle_limit = 128;
  opt.cycle_cap  = 1'000'000;
  opt.register_next_layer_inputs = true;

  ConvRunner runner(core, spine_fmt, weight_fmt, img, dram_bytes, dir, spec, opt);

  // ---- 8) Provide SpineIdFn: spine = neuron_id % (H*W) ----
  const int HW = spec.H * spec.W; // 2
  runner.SetSpineOfEntryFn([HW](const Entry& e) -> std::uint16_t {
    return static_cast<std::uint16_t>(e.neuron_id % HW);
  });

  // ---- 9) Run one layer ----
  const bool ok = runner.RunOneLayer();
  check(ok, "ConvRunner::RunOneLayer() failed");

  // After run, outputs (drained from OutputQueue via writer) are appended into dram_bytes as L+1 inputs
  img.bind(dram_bytes.data(), dram_bytes.size()); // keep RO view in sync

  // ---- 10) Verify FilterBuffer row 0 contains our seeded weight row ----
  {
    FilterBuffer::Row row{};
    const bool got = core.filter_buffer().ReadRow(0, row);
    check(got, "FilterBuffer::ReadRow(0) failed after RunOneLayer");

    for (int i = 0; i < 128; ++i) {
      check(row[i] == static_cast<int8_t>(i), "FB row0 mismatch after RunOneLayer");
    }
  }

  // ---- 11) Verify outputs registered for layer 1 and readable per spine ----
  const Range out_rng = runner.LastOutputRange();
  check(out_rng.end > out_rng.begin, "runner out range is empty");

  // dir should have inputs for L+1
  const auto in1 = dir.input_range(1);
  check(in1.begin == out_rng.begin && in1.end == out_rng.end, "dir input_range(1) not set to writer range");

  // Read back with StreamReader for both spines (0 and 1)
  {
    StreamReader rd(spine_fmt, img, dir);
    for (int s = 0; s < HW; ++s) {
      const bool ok_open = rd.open_spine(/*layer=*/1, static_cast<uint16_t>(s));
      check(ok_open, "open_spine L=1 failed");

      int total = 0;
      int segs  = 0;
      bool saw_eol = false;
      std::vector<uint8_t> line; SegmentHeader hdr{};
      while (rd.read_next(line, &hdr)) {
        ++segs;
        total += hdr.size;
        if (hdr.eol) saw_eol = true;
      }
      // We pushed 10 entries; RR by neuron_id%2 -> there will be 5 entries per spine.
      // Depending on segmenting (128 per seg), here it's 1 partial seg per spine with eol=1.
      check(segs >= 1, "expected at least 1 segment per spine");
      printf("  spine %d: segments=%d total_entries=%d eol=%d\n",
             s, segs, total, saw_eol ? 1 : 0);
      check(total == 5, "expected 5 entries per spine in L+1 inputs");
      check(saw_eol, "expected eol=1 on the last segment");
    }
  }

  std::printf("[runner-smoke] PASSED\n");
  return 0;
}
