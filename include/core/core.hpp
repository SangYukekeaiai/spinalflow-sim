// All comments are in English.
#pragma once
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <iostream>

// Common
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
#include "core/io_shadow.hpp"

// DRAM fwd-decl
namespace sf { namespace dram { class SimpleDRAM; } }

namespace sf {

struct CoreCycleStats {
  std::uint64_t load_cycles = 0;
  std::uint64_t compute_cycles = 0;
  std::uint64_t store_cycles = 0;
};

struct CoreSramStats {
  struct Component {
    std::uint64_t access_cycles = 0;
    std::uint64_t accesses = 0;
    std::uint64_t bytes = 0;
  };

  Component input_spine;
  Component filter;
  Component output_queue;

  std::uint64_t compute_load_accesses = 0;
  std::uint64_t compute_load_bytes = 0;
  std::uint64_t compute_store_accesses = 0;
  std::uint64_t compute_store_bytes = 0;

  std::uint64_t input_spine_capacity_bytes = 0;
  std::uint64_t filter_capacity_bytes = 0;
  std::uint64_t output_queue_capacity_bytes = 0;
};

class Core {
public:
  // NOTE: This is a declaration, not a definition. Do NOT write "Core::Core" here.
  explicit Core(sf::dram::SimpleDRAM* dram,
                int layer_id, int C_in, int C_out,
                int H_in, int W_in,
                int H_out, int W_out,
                int Kh, int Kw,
                int Sh, int Sw,
                int Ph, int Pw,
                float Threshold,
                int  w_bits,
                bool w_signed,
                int  w_frac_bits,
                float w_scale,
                int total_tiles,
                const std::unordered_map<std::uint64_t, std::vector<std::vector<int>>>* batches_per_hw,
                int batch_needed);


  void SetBatchesTable(const std::unordered_map<std::uint64_t,
                        std::vector<std::vector<int>>>* batches_per_hw);
  void SetTotalTiles(int total_tiles);

  // ---- Per-(h,w) prep ----
  void PrepareForSpine(int h_out, int w_out);
  void UpdatehwOut_Eachhw(int h_out, int w_out);
  void UpdateOutputSpineID_Eachhw();
  void ClearTOB_Eachhw();
  void ResetSignal_Eachhw();
  void ComputeInputSpineBatches_Eachhw();

  // ---- Per-tile prep ----
  void PrepareForTile(int tile_id);
  void ComputePEArrayOutID_EachTile(int tile_id);
  void ResetSignal_EachTile();
  std::uint32_t LoadWeightFromDram_EachTile(int tile_id);
  void LoadInputSpine_EachTile();
  void Compute_EachTile(int tile_id);

  // ---- Main step + drain ----
  bool StepOnce(int tile_id);
  void DrainAllTilesAndStore(int& drained_entries);

  // ---- Helpers ----
  bool FifosHaveData() const;
  bool TargetFifoHasSpace() const;
  bool TobEmpty() const;
  void ResetCycleStats();
  CoreCycleStats GetCycleStats() const;
  CoreSramStats GetSramStats() const;

  // Accessors
  int  layer_id() const { return layer_id_; }
  int  H_out()    const { return H_out_; }
  int  W_out()    const { return W_out_; }
  int  h_out()    const { return h_out_cur_; }
  int  w_out()    const { return w_out_cur_; }
  int  total_tiles() const { return total_tiles_; }

  const std::vector<std::vector<int>>& current_inputspine_batches() const {
    return current_inputspine_batches_;
  }

  static std::uint64_t PackHW(int h, int w) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(h)) << 32) |
           static_cast<std::uint32_t>(w);
  }


private:
  void ResetIOTracking();
  void ConsumeBlockingCycles(std::uint64_t cycles);
  void ResetSramStats();

private:
  // ---- Wiring ----
  sf::dram::SimpleDRAM* dram_ = nullptr;

  // ---- Per-layer params ----
  int layer_id_ = 0;
  int H_in_ = 0, W_in_ = 0;
  int H_out_ = 0, W_out_ = 0;
  int Kh_ = 0, Kw_ = 0;
  int Sh_ = 0, Sw_ = 0;
  int Ph_ = 0, Pw_ = 0;

  // ---- External tables (non-owning) ----
  const std::unordered_map<std::uint64_t, std::vector<std::vector<int>>>* batches_per_hw_ = nullptr;

  // ---- Value-owned subsystems ----
  IntermediateFIFO  fifos_[kNumIntermediateFifos];
  InputSpineBuffer  isb_;
  FilterBuffer      fb_;
  MinFinderBatch    mfb_;
  GlobalMerger      gm_;
  PEArray           pe_array_;
  TiledOutputBuffer tob_;
  OutputSpine       out_spine_;
  OutputSorter      sorter_;

  // ---- Per-(h,w) state ----
  int  h_out_cur_ = 0;
  int  w_out_cur_ = 0;

  bool v_tob_in_         = false;
  bool v_pe_             = false;
  bool v_mfb_            = false;
  bool compute_finished_ = false;

  bool ran_tob_in_ = false;
  bool ran_pe_     = false;
  bool ran_mfb_    = false;

  std::vector<std::vector<int>> current_inputspine_batches_;
  int batch_cursor_ = -1;
  int total_batches_needed_ = 0;

  // Per-layer tiling
  int total_tiles_ = 0;

  // Cycle counter (optional)
  std::uint64_t cycle_ = 0;

  CoreCycleStats cycle_stats_{};
  CoreSramStats sram_stats_{};
  IOShadow io_shadow_{kDefaultDramBytesPerCycle};
};

} // namespace sf
