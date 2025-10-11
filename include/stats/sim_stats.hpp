// All comments are in English.
#pragma once
#include <cstdint>

namespace sf {

struct StageStats {
  uint64_t ran = 0;
  uint64_t gated_off = 0;
  uint64_t eligible_but_noop = 0;
};

struct LayerCycleStats {
  // Input loads
  uint64_t preload_input_cycles       = 0;
  uint64_t load_input_cycles_in_step  = 0;

  // Weights
  uint64_t weight_load_cycles         = 0;

  // Output
  uint64_t output_drain_cycles        = 0;
  uint64_t output_store_cycles        = 0;

  // Per-step accounting
  uint64_t step_ticks                 = 0;
  uint64_t step_cycles_total          = 0;
  uint64_t step_extra_memload_cycles  = 0;

  // Per-stage counters
  StageStats tob_in;   // Stage 0
  StageStats pe;       // Stage 1
  StageStats mfb;      // Stage 2
  StageStats isb_ld;   // Stage 3

  void ResetSite() {
    preload_input_cycles = 0;
    load_input_cycles_in_step = 0;
    weight_load_cycles = 0;
    output_drain_cycles = 0;
    output_store_cycles = 0;
    step_ticks = 0;
    step_cycles_total = 0;
    step_extra_memload_cycles = 0;
    tob_in = {};
    pe     = {};
    mfb    = {};
    isb_ld = {};
  }
};

// Helper: accumulate StageStats
inline void AccumulateStage(StageStats& dst, const StageStats& src) {
  dst.ran               += src.ran;
  dst.gated_off         += src.gated_off;
  dst.eligible_but_noop += src.eligible_but_noop;
}

// Helper: accumulate LayerCycleStats (site -> layer)
inline void AccumulateLayerStats(LayerCycleStats& dst, const LayerCycleStats& src) {
  dst.preload_input_cycles      += src.preload_input_cycles;
  dst.load_input_cycles_in_step += src.load_input_cycles_in_step;
  dst.weight_load_cycles        += src.weight_load_cycles;
  dst.output_drain_cycles       += src.output_drain_cycles;
  dst.output_store_cycles       += src.output_store_cycles;

  dst.step_ticks                += src.step_ticks;
  dst.step_cycles_total         += src.step_cycles_total;
  dst.step_extra_memload_cycles += src.step_extra_memload_cycles;

  AccumulateStage(dst.tob_in,  src.tob_in);
  AccumulateStage(dst.pe,      src.pe);
  AccumulateStage(dst.mfb,     src.mfb);
  AccumulateStage(dst.isb_ld,  src.isb_ld);
}

} // namespace sf
