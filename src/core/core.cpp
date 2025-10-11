// All comments are in English.

#include "core/core.hpp"

#include <stdexcept>
#include <algorithm>
#include <iostream>

namespace sf {

Core::Core(sf::dram::SimpleDRAM* dram,
           FilterBuffer* fb,
           InputSpineBuffer* isb)
  : dram_(dram),
    fb_(fb),
    isb_(isb),
    mfb_(isb_, fifos_),    // constructed with ISB* and FIFO array
    gm_(fifos_, mfb_),     // GM sees FIFOs and MFB
    pe_array_(gm_),        // PE array uses GM
    tob_(pe_array_),       // TOB aggregates spikes per tile (no tile_id inside)
    out_spine_(dram_, /*spine_id=*/0, kOutputSpineMaxEntries)
{
  // Bind OutputSorter to current TOB and OutputSpine.
  sorter_ = std::make_unique<OutputSorter>(&tob_, &out_spine_);

  // Initial valids
  v_tob_in_ = true;
  v_pe_     = false;
  v_mfb_    = false;
}
// Set weight quantization params for the PEArray.
void Core::SetPEsWeightParamsAndThres(float threshold, int w_bits, bool w_signed, int w_frac_bits, float w_scale) {
  pe_array_.SetWeightParamsAndThres(threshold, w_bits, w_signed, w_frac_bits, w_scale);
}
void Core::SetSpineContext(int layer_id, int h_out, int w_out, int W_out) {
  layer_id_ = layer_id;
  h_out_    = h_out;
  w_out_    = w_out;
  W_out_    = W_out;

  // Rebind OutputSpine to new spine id.
  const int spine_id = h_out_ * W_out_ + w_out_;
  out_spine_ = OutputSpine(dram_, spine_id, kOutputSpineMaxEntries);

  // Rebind sorter to current TOB and refreshed OutputSpine.
  sorter_ = std::make_unique<OutputSorter>(&tob_, &out_spine_);
}

void Core::ConfigureTiles(int C_out) {
  if (C_out <= 0) {
    throw std::invalid_argument("Core::ConfigureTiles: C_out must be positive.");
  }
  total_tiles_ = static_cast<int>(
    (C_out + static_cast<int>(kNumPE) - 1) / static_cast<int>(kNumPE));
  // std::cout<<"Core::ConfigureTiles: total_tiles = " << std::to_string(total_tiles_) << "\n";
  if (total_tiles_ <= 0) {
    throw std::runtime_error("Core::ConfigureTiles: computed total_tiles <= 0.");
  }
  if (total_tiles_ > static_cast<int>(kTilesPerSpine)) {
    throw std::runtime_error("Core::ConfigureTiles: total_tiles exceeds kTilesPerSpine capacity.");
  }

  // Fresh site: clear TOB per-tile buffers to reuse the object.
  tob_.ClearAll();

  // Rebind sorter (tob_ unchanged, out_spine_ is current site).
  sorter_ = std::make_unique<OutputSorter>(&tob_, &out_spine_);

  // Reset valids for the upcoming compute.
  v_tob_in_ = true;
  v_pe_     = false;
  v_mfb_    = false;

  // Clear progress flags.
  compute_finished_ = false;
}

void Core::BindTileBatches(const std::vector<std::vector<int>>* batches) {
  spine_batches_ = batches;
  if (!spine_batches_) {
    throw std::invalid_argument("Core::BindTileBatches: spine_batches must not be null.");
  }
  total_batches_needed_ = static_cast<int>(spine_batches_->size());
  batch_cursor_ = 0;

  // Initialize valids (start with MFB only if ISB has data and target FIFO has room).
  v_tob_in_ = true;
  v_pe_     = false;
  v_mfb_    = (!isb_->AllEmpty()) && TargetFifoHasSpace();
}

void Core::PreloadFirstBatch(uint64_t* out_cycles) {
  if (!spine_batches_ || spine_batches_->empty()) {
    throw std::runtime_error("Core::PreloadFirstBatch: spine_batches not bound or empty.");
  }
  uint64_t preload_cycles = 0;
  // Dispatch logical ids to physical ISBs via DRAM.
  isb_->PreloadFirstBatch((*spine_batches_)[0], layer_id_, &preload_cycles);
  batch_cursor_ = 0;
  cycle_ += preload_cycles;
  if (stats_) stats_->preload_input_cycles += preload_cycles;
  if (out_cycles) *out_cycles = preload_cycles;
}

std::uint32_t Core::LoadWeightFromDram(std::uint32_t layer_id, std::uint32_t tile_id, uint64_t* out_cycles) {
  uint64_t cycles = 0;
  const std::uint32_t bytes = fb_->LoadWeightFromDram( total_tiles_ ,tile_id, layer_id, &cycles);
  cycle_ += cycles;
  if (stats_) stats_->weight_load_cycles += cycles;
  if (out_cycles) *out_cycles = cycles;

  return bytes;
}

void Core::InitPEsOutputNIDBeforeLoop(int tile_idx) {
  // Basic guards to avoid programming PEs with invalid parameters.
  if (total_tiles_ <= 0) {
    throw std::runtime_error("Core::InitPEsBeforeLoop: total_tiles_ not configured. Call ConfigureTiles(C_out) first.");
  }
  if (tile_idx < 0 || tile_idx >= total_tiles_) {
    throw std::out_of_range("Core::InitPEsBeforeLoop: tile_idx out of range.");
  }
  if (W_out_ <= 0) {
    throw std::runtime_error("Core::InitPEsBeforeLoop: W_out_ not set. Call SetSpineContext(...) first.");
  }
  if (h_out_ < 0 || w_out_ < 0) {
    throw std::runtime_error("Core::InitPEsBeforeLoop: invalid (h_out_, w_out_).");
  }

  // Fresh tile: allow the outer loop to enter StepOnce again and re-prime valids.
  compute_finished_ = false;
  v_tob_in_ = true;
  v_pe_     = false;
  v_mfb_    = (!isb_->AllEmpty()) && TargetFifoHasSpace();

  // Forward to PEArray's helper. PEArray will:
  // - Compute out_id for each PE = base_pos + tile_offset + pe_idx
  // - Register per-PE output id and set threshold
  // - Clear its internal out_spike_entries_ for a fresh tile
  pe_array_.InitPEsOutputNIDBeforeLoop(
      /*total_tiles =*/ total_tiles_,
      /*tile_idx    =*/ tile_idx,
      /*h           =*/ h_out_,
      /*w           =*/ w_out_,
      /*W           =*/ W_out_);
}
// All comments are in English.
bool Core::StepOnce(int tile_id) {
  if (tile_id < 0 || tile_id >= total_tiles_) {
    throw std::out_of_range("Core::StepOnce: tile_id out of range.");
  }

  // One StepOnce call equals one synchronous tick.
  uint64_t step_extra_cycles = 0;       // extra cycles charged inside this step (e.g., ISB load)
  if (stats_) stats_->step_ticks += 1;  // count this tick

  // ---------------------------
  // Stage 0 – TiledOutputBuffer
  // ---------------------------
  ran_tob_in_ = v_tob_in_ ? tob_.run(tile_id) : false;
  if (stats_) {
    if (!v_tob_in_)                 stats_->tob_in.gated_off++;
    else if (!ran_tob_in_)          stats_->tob_in.eligible_but_noop++;
    else                            stats_->tob_in.ran++;
  }

  // ---------------------------
  // Stage 1 – PEArray
  // ---------------------------
  ran_pe_ = v_pe_ ? pe_array_.run(*fb_) : false;
  if (stats_) {
    if (!v_pe_)                     stats_->pe.gated_off++;
    else if (!ran_pe_)              stats_->pe.eligible_but_noop++;
    else                            stats_->pe.ran++;
  }

  // ---------------------------
  // Stage 2 – MinFinderBatch
  // ---------------------------
  ran_mfb_ = v_mfb_ ? mfb_.run(batch_cursor_, total_batches_needed_) : false;
  if (stats_) {
    if (!v_mfb_)                    stats_->mfb.gated_off++;
    else if (!ran_mfb_)             stats_->mfb.eligible_but_noop++;
    else                            stats_->mfb.ran++;
  }

  // ---------------------------
  // Stage 3 – ISB batch load (if buffers empty and more batches remain)
  // ---------------------------
  uint64_t load_cycles = 0; // must be filled by isb_->run(..., &load_cycles)
  const bool can_try_isb_load = isb_->AllEmpty() && (batch_cursor_ + 1 < total_batches_needed_);
  if (!can_try_isb_load) {
    if (stats_) stats_->isb_ld.gated_off++;
  } else {
    ++batch_cursor_;
    // IMPORTANT: use the out_cycles overload so load_cycles is populated.
    const bool loaded = isb_->run((*spine_batches_)[batch_cursor_],
                                  layer_id_,
                                  batch_cursor_,
                                  total_batches_needed_,
                                  &load_cycles);
    if (stats_) {
      if (!loaded) stats_->isb_ld.eligible_but_noop++;
      else         stats_->isb_ld.ran++;
    }
    if (loaded) {
      // Charge the memory load cycles to this step and to the global clock.
      step_extra_cycles += load_cycles;
      cycle_ += load_cycles;
      if (stats_) {
        // This belongs to the "in-step" input load bucket, not preload.
        stats_->load_input_cycles_in_step += load_cycles;
        stats_->step_extra_memload_cycles += load_cycles;
      }
    }
  }

  // Compute next valids (hard backpressure + FIFO capacity for MFB).
  const bool stall = tob_.stall_next_cycle();  // replaces cooldown semantics

  const auto& pe_slots = pe_array_.out_spike_entries();
  bool pe_hasout = false;
  for (const auto& s : pe_slots) {
    if (s.has_value()) { pe_hasout = true; break; }
  }

  const bool fifo_has   = FifosHaveData();
  const bool isb_has    = !isb_->AllEmpty();
  const bool fifo_space = TargetFifoHasSpace();

  // Allow TOB to run every cycle so it can drain its local per-PE FIFOs
  // even if the PEArray has no new outputs this cycle.
  const bool v_tob_in_next = true;

  // Stall from TOB blocks PE/MFB ingestion for the next cycle.
  const bool v_pe_next  = (!stall) && fifo_has;
  const bool v_mfb_next = (!stall) && isb_has && fifo_space;

  v_tob_in_ = v_tob_in_next;
  v_pe_     = v_pe_next;
  v_mfb_    = v_mfb_next;

  // Finish condition for compute of THIS tile_id (no cooldown anymore).
  // Note: this ignores TOB's internal FIFOs; TOB will keep draining them since
  // v_tob_in_next is always true.
  compute_finished_ = (!stall) && !fifo_has && !pe_hasout && !isb_has;

  // End-of-step: add the synchronous tick + any extra cycles charged above.
  cycle_ += 1;
  if (stats_) stats_->step_cycles_total += (1 + step_extra_cycles);

  return (ran_tob_in_ || ran_pe_ || ran_mfb_);
}


void Core::DrainAllTilesAndStore(int & drained_entries) {
   
  if (!sorter_) {
    sorter_ = std::make_unique<OutputSorter>(&tob_, &out_spine_);
  }
  // sorter_->AnnouncedSize();
  const uint64_t kCyclesPerEntryDrain = 1;
  // Drain tile buffers to OutputSpine, one entry per Sort() call.
  uint64_t drain_sort_calls = 0;
  while (sorter_->Sort()) {
    ++drain_sort_calls;
  }
  // Store to DRAM (throws on failure). Clears OutputSpine on success.
  

  drained_entries += static_cast<int>(out_spine_.size());
  if(drain_sort_calls != out_spine_.size()){
    std::cout << "Warning: drain_sort_calls(" << drain_sort_calls << ") != out_spine_size(" << out_spine_.size() << ")\n";
  }
  uint64_t store_cycles = 0;
  out_spine_.StoreOutputSpineToDRAM(static_cast<std::uint32_t>(layer_id_), &store_cycles);

  // Accumulate cycles to core_ clock and stats.
  const uint64_t drain_cycles = drain_sort_calls * kCyclesPerEntryDrain;
  cycle_ += drain_cycles + store_cycles;

  if (stats_) {
    stats_->output_drain_cycles += drain_cycles;
    stats_->output_store_cycles += store_cycles;
  }

}

bool Core::FifosHaveData() const {
  for (std::size_t i = 0; i < kNumIntermediateFifos; ++i) {
    if (!fifos_[i].empty()) return true;
  }
  return false;
}

bool Core::TargetFifoHasSpace() const {
  if (!spine_batches_) return false;
  if (batch_cursor_ < 0 || batch_cursor_ >= static_cast<int>(kNumIntermediateFifos)) {
    return false;
  }
  return !fifos_[static_cast<std::size_t>(batch_cursor_)].full();
}

bool Core::TobEmpty() const {
  Entry tmp{};
  const int limit = std::max(0, total_tiles_);
  for (int i = 0; i < limit; ++i) {
    if (tob_.PeekTileHead(static_cast<std::size_t>(i), tmp)) {
      return false;
    }
  }
  return true;
}

} // namespace sf
