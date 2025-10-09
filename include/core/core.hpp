#pragma once
// All comments are in English.

#include <cstdint>
#include <cstddef>
#include <memory>
#include <vector>

#include "common/constants.hpp"
#include "common/entry.hpp"

// Subsystems
#include "arch/filter_buffer.hpp"
#include "arch/input_spine_buffer.hpp"
#include "arch/intermediate_fifo.hpp"
#include "arch/min_finder_batch.hpp"
#include "arch/global_merger.hpp"
#include "arch/pe_array.hpp"
#include "arch/tiled_output_buffer.hpp"
#include "arch/output_spine.hpp"
#include "arch/output_sorter.hpp"

namespace sf {

/**
 * Core
 *
 * Scheduler and glue between: ISB -> MinFinderBatch -> FIFOs -> GM -> PEArray
 * -> TiledOutputBuffer -> OutputSorter -> OutputSpine.
 *
 * Design notes (per latest spec):
 * - No StartTile(): TOB does not carry a tile_id. The caller passes tile_id to StepOnce(tile_id).
 * - Sorting/draining to OutputSpine must happen only after ALL tiles have finished compute.
 * - OutputSorter remains the original single-entry "Sort()" popper scanning kTilesPerSpine heads.
 * - total_tiles = C_out / kNumPE and must be <= kTilesPerSpine (guarded).
 * - conv_layer (or driver) is responsible for calling FB::Configure(...) and FB::Update(h_out,w_out).
 */
class Core {
public:
  Core(sf::dram::SimpleDRAM* dram,
       FilterBuffer* fb,
       InputSpineBuffer* isb);

  // ----- High-level context setters -----

  // Set weight quantization params for the PEArray.
  void SetPEsWeightParamsAndThres(float threshold, int w_bits, bool w_signed, int w_frac_bits, float w_scale);
  /**
   * Initialize PEArray before entering the per-tile compute loop.
   * - Programs each PE's output neuron id for this (h_out_, w_out_, tile_idx).
   * - Sets the firing threshold for all PEs.
   *
   * Pre-conditions:
   *   - ConfigureTiles(C_out) has been called (so total_tiles_ > 0).
   *   - SetSpineContext(...) has been called (valid h_out_, w_out_, W_out_).
   *   - 0 <= tile_idx < total_tiles_.
   */
  void InitPEsOutputNIDBeforeLoop(int tile_idx);
  // Set layer id and output-spine coordinates (h_out, w_out, W_out).
  // Rebinds OutputSpine's spine_id accordingly.
  void SetSpineContext(int layer_id, int h_out, int w_out, int W_out);

  // Compute total tiles from C_out and kNumPE; clears TOB buffers for a fresh site.
  // Also rebinds/refreshes the sorter to (tob_, out_spine_).
  void ConfigureTiles(int C_out);

  // Bind spine_batches pointer for this (h_out, w_out) site.
  // total_batches_needed is derived as spine_batches->size() or kernel-slot bound.
  void BindTileBatches(const std::vector<std::vector<int>>* batches);

  // Pre-load the first batch into ISB; 'spine_batches' must be bound.
  void PreloadFirstBatch();

  // ----- Cycle-level stepping -----

  // Execute one scheduler step (one cycle) for the provided tile_id.
  // Returns true if any stage ran; false if completely idle this cycle.
  bool StepOnce(int tile_id);

  // ----- Global drain & store (to be called only after all tiles computed) -----

  // Drain all per-tile buffers (inside TOB) into OutputSpine using the sorter,
  // then store OutputSpine to DRAM.
  void DrainAllTilesAndStore(int & drained_entries);

  // ----- Status / helpers -----
  bool FinishedCompute() const { return compute_finished_; }
  int  total_tiles() const { return total_tiles_; }

private:
  // Helpers
  bool FifosHaveData() const;
  bool TargetFifoHasSpace() const;
  bool TobEmpty() const;

private:
  // -------- External context (from conv_layer) --------
  int layer_id_ = 0;
  int h_out_ = 0, w_out_ = 0, W_out_ = 0;

  // -------- Tile control (Core-owned) --------
  int total_tiles_ = 0;   // C_out / kNumPE  (must be <= kTilesPerSpine)

  // -------- Batching --------
  const std::vector<std::vector<int>>* spine_batches_ = nullptr;
  int total_batches_needed_ = 0;
  int batch_cursor_ = 0;

  // -------- Subsystems / wiring --------
  sf::dram::SimpleDRAM* dram_ = nullptr;  // non-owning
  FilterBuffer*         fb_   = nullptr;  // non-owning
  InputSpineBuffer*     isb_  = nullptr;  // non-owning

  IntermediateFIFO fifos_[kNumIntermediateFifos]; // owned
  MinFinderBatch   mfb_;   // uses isb_ and fifos_
  GlobalMerger     gm_;    // uses fifos_ and mfb_
  PEArray          pe_array_;

  // Single TOB that holds all per-tile buffers; no tile_id member inside.
  TiledOutputBuffer tob_;

  OutputSpine                          out_spine_;
  std::unique_ptr<OutputSorter>        sorter_;  // scans all tile heads in TOB

  // -------- Per-cycle valid / ran (Core-owned) --------
  bool v_tob_in_ = true;   // Stage 0: TiledOutputBuffer ingress
  bool v_pe_     = false;  // Stage 1: PEArray.run
  bool v_mfb_    = false;  // Stage 2: MinFinderBatch.run

  bool ran_tob_in_ = false;
  bool ran_pe_     = false;
  bool ran_mfb_    = false;

  // -------- Progress (for the last StepOnce(tile_id)) --------
  bool      compute_finished_ = false;  // compute loop done for the tile of the last StepOnce
  uint64_t  cycle_ = 0;
};

} // namespace sf
