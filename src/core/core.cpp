// All comments are in English.
#include "core/core.hpp"

namespace sf {

using sf::dram::SimpleDRAM;

Core::Core(SimpleDRAM* dram,
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
           int batch_needed)
  : dram_(dram),
    // Value members are default-constructed; wire dependencies via their constructors if available.
    isb_(dram),                 // ISB needs DRAM
    fb_(),                      // FB configured below
    mfb_(&isb_, fifos_),        // MFB sees ISB and FIFOs
    gm_(fifos_, mfb_),          // GM sees FIFOs and MFB
    pe_array_(gm_),             // PE array uses GM
    tob_(pe_array_),            // TOB aggregates per-PE spikes by tile
    out_spine_(dram_, kOutputSpineMaxEntries),
    sorter_(&tob_, &out_spine_),
    layer_id_(layer_id),
    H_in_(H_in), W_in_(W_in),
    H_out_(H_out), W_out_(W_out),
    Kh_(Kh), Kw_(Kw),
    Sh_(Sh), Sw_(Sw),
    Ph_(Ph), Pw_(Pw),
    batches_per_hw_(batches_per_hw),   // FIX: was a typo "batcches_per_hw_"
    total_tiles_(total_tiles),
    total_batches_needed_(batch_needed)
{
  if (!dram_) {
    throw std::invalid_argument("Core: dram pointer must not be null.");
  }

  // Configure static FB params once for the layer.
  fb_.Configure(C_in, W_in, Kh, Kw, Sh, Sw, Ph, Pw, dram_);

  // Program PE weight/threshold params once.
  pe_array_.SetWeightParamsAndThres(Threshold, w_bits, w_signed, w_frac_bits, w_scale);

}



void Core::SetBatchesTable(const std::unordered_map<std::uint64_t,
                             std::vector<std::vector<int>>>* batches_per_hw)
{
  batches_per_hw_ = batches_per_hw;
}

void Core::SetTotalTiles(int total_tiles)
{
  if (total_tiles <= 0) {
    throw std::invalid_argument("Core::SetTotalTiles: total_tiles must be > 0.");
  }
  total_tiles_ = total_tiles;
}

void Core::PrepareForSpine(int h_out, int w_out)
{
  UpdatehwOut_Eachhw(h_out, w_out);
  UpdateOutputSpineID_Eachhw();
  ClearTOB_Eachhw();
  ResetSignal_Eachhw();
  ComputeInputSpineBatches_Eachhw();
}

void Core::UpdatehwOut_Eachhw(int h_out, int w_out)
{
  if (h_out < 0 || h_out >= H_out_ || w_out < 0 || w_out >= W_out_) {
    throw std::out_of_range("Core::Update_Eachhw: (h_out, w_out) is out of range.");
  }
  h_out_cur_ = h_out;
  w_out_cur_ = w_out;
  fb_.Update(h_out_cur_, w_out_cur_);
}

void Core::UpdateOutputSpineID_Eachhw()
{
  const int spine_id = h_out_cur_ * W_out_ + w_out_cur_;
  out_spine_.SetSpineID(spine_id);
}

void Core::ClearTOB_Eachhw()
{
  tob_.ClearAll();
}

void Core::ResetSignal_Eachhw()
{
  v_tob_in_         = false;
  v_pe_             = false;
  v_mfb_            = false;
  compute_finished_ = false;
}

void Core::ComputeInputSpineBatches_Eachhw()
{
  current_inputspine_batches_.clear();
  batch_cursor_ = -1;

  if (!batches_per_hw_) {
    std::cout << "[Core] batches_per_hw_ not set; no input spine batches.\n";
    total_batches_needed_ = 0;
    return;
  }

  const std::uint64_t key = PackHW(h_out_cur_, w_out_cur_);
  auto it = batches_per_hw_->find(key);
  if (it == batches_per_hw_->end()) {
    total_batches_needed_ = 0;
    return;
  }
  current_inputspine_batches_ = it->second; // copy small vectors
  total_batches_needed_ = static_cast<int>(current_inputspine_batches_.size());
}

// ==================== Per-tile sequence ====================

void Core::PrepareForTile(int tile_id)
{
  ComputePEArrayOutID_EachTile(tile_id);
  ResetSignal_EachTile();
  LoadWeightFromDram_EachTile(tile_id);
  LoadInputSpine_EachTile();
}

void Core::ComputePEArrayOutID_EachTile(int tile_id)
{
  if (total_tiles_ <= 0) {
    throw std::runtime_error("Core::ComputePEArrayOutID_EachTile: total_tiles_ not set.");
  }
  if (tile_id < 0 || tile_id >= total_tiles_) {
    throw std::out_of_range("Core::ComputePEArrayOutID_EachTile: tile_id out of range.");
  }
  if (W_out_ <= 0) {
    throw std::runtime_error("Core::ComputePEArrayOutID_EachTile: W_out_ not set.");
  }

  compute_finished_ = false;

  pe_array_.InitPEsOutputNIDBeforeLoop(/*total_tiles=*/ total_tiles_,
                                       /*tile_idx   =*/ tile_id,
                                       /*h         =*/ h_out_cur_,
                                       /*w         =*/ w_out_cur_,
                                       /*W         =*/ W_out_);
}

void Core::ResetSignal_EachTile()
{
  compute_finished_ = false;
  v_tob_in_ = true;
  v_pe_     = false;

  const bool isb_has_data = !isb_.AllEmpty();
  v_mfb_ = isb_has_data && TargetFifoHasSpace();
}

std::uint32_t Core::LoadWeightFromDram_EachTile(int tile_id)
{
  if (total_tiles_ <= 0) {
    throw std::runtime_error("Core::LoadWeightFromDram_EachTile: total_tiles_ not set.");
  }
  if (tile_id < 0 || tile_id >= total_tiles_) {
    throw std::out_of_range("Core::LoadWeightFromDram_EachTile: tile_id out of range.");
  }

  const std::uint32_t bytes = fb_.LoadWeightFromDram(total_tiles_, tile_id, layer_id_);
  return bytes;
}

void Core::LoadInputSpine_EachTile()
{
  if (current_inputspine_batches_.empty()) {
    throw std::runtime_error("Core::LoadInputSpine_EachTile: no batches for current (h,w).");
  }
  isb_.PreloadFirstBatch(current_inputspine_batches_[0], layer_id_);
  batch_cursor_ = 0;
}

void Core::Compute_EachTile(int tile_id)
{
  if (tile_id < 0 || tile_id >= total_tiles_) {
    throw std::out_of_range("Core::ComputeTiles: tile_id out of range.");
  }
  if (total_batches_needed_ <= 0) {
    return;
  }
  if (batch_cursor_ < 0) {
    throw std::runtime_error("Core::ComputeTiles: first batch not preloaded; call LoadInputSpine_EachTile() first.");
  }

  for (int b = batch_cursor_; b < total_batches_needed_; ++b) {
    // Run the compute loop for the current batch.
    compute_finished_ = false;
    while (!compute_finished_) {
      StepOnce(tile_id);
    }

    const bool has_next = (b + 1 < total_batches_needed_);
    if (has_next) {
      const int next_b = b + 1;
      const bool loaded = isb_.run(
          current_inputspine_batches_[static_cast<std::size_t>(next_b)],
          layer_id_,
          next_b,
          total_batches_needed_);
      (void)loaded;
      batch_cursor_ = next_b;
    }
  }
}

// ==================== StepOnce & Drain ====================

bool Core::StepOnce(int tile_id) {
  if (tile_id < 0 || tile_id >= total_tiles_) {
    throw std::out_of_range("Core::StepOnce: tile_id out of range.");
  }

  // ---------------------------
  // Stage 0 – TiledOutputBuffer
  // ---------------------------
  ran_tob_in_ = v_tob_in_ ? tob_.run(static_cast<std::size_t>(tile_id)) : false;

  // ---------------------------
  // Stage 1 – PEArray
  // ---------------------------
  ran_pe_ = v_pe_ ? pe_array_.run(fb_) : false;

  // ---------------------------
  // Stage 2 – MinFinderBatch
  // ---------------------------
  ran_mfb_ = v_mfb_ ? mfb_.run(batch_cursor_, total_batches_needed_) : false;

  // ---------------------------
  // NOTE: Stage 3 (ISB load of next batch) has been REMOVED from StepOnce.
  // Batch progression is now handled by ComputeTiles().
  // ---------------------------

  // Compute next valids (hard backpressure + FIFO capacity for MFB).
  const bool stall = tob_.stall_next_cycle();  // replaces cooldown semantics

  const auto& pe_slots = pe_array_.out_spike_entries();
  bool pe_hasout = false;
  for (const auto& s : pe_slots) {
    if (s.has_value()) { pe_hasout = true; break; }
  }

  const bool fifo_has   = FifosHaveData();
  const bool isb_has    = !isb_.AllEmpty();
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

  // Finish condition for compute of THIS batch for THIS tile.
  // TOB will keep draining since v_tob_in_next is always true.
  compute_finished_ = (!stall) && !fifo_has && !pe_hasout && !isb_has;

  // End-of-step: add the synchronous tick.
  cycle_ += 1;

  return (ran_tob_in_ || ran_pe_ || ran_mfb_);
}
void Core::DrainAllTilesAndStore(int & drained_entries) {
  const uint64_t kCyclesPerEntryDrain = 1;
  uint64_t drain_sort_calls = 0;

  while (sorter_.Sort()) {
    ++drain_sort_calls;
  }

  drained_entries += static_cast<int>(out_spine_.size());
  if (drain_sort_calls != out_spine_.size()) {
    std::cout << "Warning: drain_sort_calls(" << drain_sort_calls
              << ") != out_spine_size(" << out_spine_.size() << ")\n";
  }

  out_spine_.StoreOutputSpineToDRAM(static_cast<std::uint32_t>(layer_id_));

  const uint64_t drain_cycles = drain_sort_calls * kCyclesPerEntryDrain;
  drained_entries += static_cast<int>(out_spine_.size());
}

bool Core::FifosHaveData() const {
  for (std::size_t i = 0; i < kNumIntermediateFifos; ++i) {
    if (!fifos_[i].empty()) return true;
  }
  return false;
}

bool Core::TargetFifoHasSpace() const {
  if (current_inputspine_batches_.empty()) return false;
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
