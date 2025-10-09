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

void Core::PreloadFirstBatch() {
  if (!spine_batches_ || spine_batches_->empty()) {
    throw std::runtime_error("Core::PreloadFirstBatch: spine_batches not bound or empty.");
  }
  // Dispatch logical ids to physical ISBs via DRAM.
  isb_->PreloadFirstBatch((*spine_batches_)[0], layer_id_);
  batch_cursor_ = 0;
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
bool Core::StepOnce(int tile_id) {
  if (tile_id < 0 || tile_id >= total_tiles_) {
    throw std::out_of_range("Core::StepOnce: tile_id out of range.");
  }

  // Stage 0 – TiledOutputBuffer: copy PE outputs to the per-tile buffer or decrement cooldown.
  ran_tob_in_ = v_tob_in_ ? tob_.run(tile_id) : false;

  // Stage 1 – PEArray: compute if allowed; will fetch (GM entry + FB row) internally.
  ran_pe_ = v_pe_ ? pe_array_.run(*fb_) : false;

  // Stage 2 – MinFinderBatch: drain ISB -> FIFOs (pass batch cursor/total by value).
  ran_mfb_ = v_mfb_ ? mfb_.run(batch_cursor_, total_batches_needed_) : false;

  // Stage 3 – Load next batch if needed.
  if (isb_->AllEmpty() && (batch_cursor_ + 1 < total_batches_needed_)) {
    ++batch_cursor_;
    // Load the next batch into ISB.
    isb_->run((*spine_batches_)[batch_cursor_], layer_id_, batch_cursor_, total_batches_needed_);
  }

  // Compute next valids (hard backpressure + FIFO capacity for MFB).
  // NEW: TOB exposes a boolean stall flag for the *next* cycle.
  const bool stall = tob_.stall_next_cycle();  // replaces cooldown semantics

  // NEW: PE outputs are a fixed array of optionals; detect "any has value".
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

  ++cycle_;
  return (ran_tob_in_ || ran_pe_ || ran_mfb_);
}

void Core::DrainAllTilesAndStore(int & drained_entries) {
   
  if (!sorter_) {
    sorter_ = std::make_unique<OutputSorter>(&tob_, &out_spine_);
  }
  // Drain tile buffers to OutputSpine, one entry per Sort() call.
  while (sorter_->Sort()) {
    // keep popping one-by-one
  }
  // Store to DRAM (throws on failure). Clears OutputSpine on success.
  

  drained_entries += static_cast<int>(out_spine_.size());
  // std::cout << "callee addr=" << &drained_entries << "\n";
  out_spine_.StoreOutputSpineToDRAM(static_cast<std::uint32_t>(layer_id_));

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
