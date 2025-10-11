// All comments are in English.

#include "arch/filter_buffer.hpp"
#include <iostream>

namespace sf {

void FilterBuffer::Configure(int C_in, int W_in,
                             int Kh, int Kw,
                             int Sh, int Sw,
                             int Ph, int Pw,
                             sf::dram::SimpleDRAM* dram_ptr) {
  if (C_in <= 0 || W_in <= 0 || Kh <= 0 || Kw <= 0 || Sh <= 0 || Sw <= 0) {
    throw std::invalid_argument("FilterBuffer::Configure: non-positive dimension/stride.");
  }
  C_in_ = C_in;
  W_in_ = W_in;
  K_h_  = Kh;
  K_w_  = Kw;
  S_h_  = Sh;
  S_w_  = Sw;
  P_h_  = Ph;
  P_w_  = Pw;
  dram_ = dram_ptr;
  // Reset ownership on (re)configuration
  ClearAllOwnership();

  // Optional: zero the storage (not strictly required)
  for (auto& r : rows_) r.fill(0);
}

void FilterBuffer::Update(int h_out, int w_out) {
  h_out_cur_ = h_out;
  w_out_cur_ = w_out;
}

int FilterBuffer::ComputeRowId(std::uint32_t neuron_id) const {
  // Guard configuration.
  if (C_in_ <= 0 || W_in_ <= 0 || K_h_ <= 0 || K_w_ <= 0) return -1;

  // 1) Input channel index from neuron_id and C_in_.
  const int c_in = static_cast<int>(neuron_id % static_cast<std::uint32_t>(C_in_));

  // 2) Decode input spatial (h_in, w_in) from neuron_id using W_in_.
  //    neuron_id = C_in * (h_in * W_in + w_in) + c_in
  const std::uint32_t pos_in = static_cast<std::uint32_t>(neuron_id / static_cast<std::uint32_t>(C_in_));
  const int h_in = static_cast<int>(pos_in / static_cast<std::uint32_t>(W_in_));
  const int w_in = static_cast<int>(pos_in % static_cast<std::uint32_t>(W_in_));

  // 3) Compute kernel offsets (r,c) using current output site and stride/padding (members).
  //    r = h_in - (h_out_cur * S_h - P_h)
  //    c = w_in - (w_out_cur * S_w - P_w)
  const int r = h_in - (h_out_cur_ * S_h_ - P_h_);
  const int c = w_in - (w_out_cur_ * S_w_ - P_w_);

  // 4) Check (r,c) within the kernel window.
  if (r < 0 || r >= K_h_ || c < 0 || c >= K_w_) {
    std::cout << "ComputeRowId: neuron_id=" << neuron_id
              << " maps to (c_in=" << c_in << ", r=" << r << ", c=" << c << ") outside kernel window\n";
    return -1; // invalid/padded tap
  }

  // 5) Flatten (c_in, r, c) -> row_id
  const long long row_id_ll =
      (static_cast<long long>(c_in) * K_h_ + r) * K_w_ + c;

  if (row_id_ll < 0 || row_id_ll >= static_cast<long long>(kFilterRows)) {
    std::cout << "ComputeRowId: neuron_id=" << neuron_id
              << " maps to out-of-bounds row_id=" << row_id_ll
              << " (c_in=" << c_in << ", r=" << r << ", c=" << c
              << ", K_h=" << K_h_ << ", K_w=" << K_w_ <<
            ")\n";
    return -1; // storage bound check (fixed capacity)
  }

  return static_cast<int>(row_id_ll);
}

FilterBuffer::Row FilterBuffer::GetRow(int row_id) const {
  const int rpt = RowsPerTile();
  if (rpt <= 0) throw std::logic_error("GetRow: invalid rows-per-tile (configure layer first).");
  if (row_id < 0 || row_id >= rpt) throw std::out_of_range("GetRow: row_id out of range for active tile.");
  const uint32_t base = ActiveBaseRow();
  const uint32_t idx  = base + static_cast<uint32_t>(row_id);
  if (idx >= kFilterRows) throw std::out_of_range("GetRow: computed row index exceeds buffer.");
  return rows_[idx];
}

std::uint32_t FilterBuffer::LoadWeightFromDram(std::uint32_t total_tiles,
                                               std::uint32_t tile_id,
                                               std::uint32_t layer_id,
                                               uint64_t* out_cycles) {
  if (out_cycles) *out_cycles = 0;
  if (!dram_) {
    throw std::runtime_error("FilterBuffer::LoadWeightFromDram: DRAM pointer is null.");
  }
  if (total_tiles == 0) {
    throw std::invalid_argument("FilterBuffer::LoadWeightFromDram: total_tiles must be > 0.");
  }

  // If already resident: make it active and return.
  if (owned_tile_id_.count(tile_id)) {
    active_tile_id_ = tile_id; // just switch active tile
    return 0;                  // no DRAM access
  }

  // Compute rows per tile and capacity checks.
  const int rows_per_tile = RowsPerTile();
  if (rows_per_tile <= 0) {
    throw std::logic_error("FilterBuffer::LoadWeightFromDram: rows_per_tile <= 0 (configure layer first).");
  }
  if (kFilterRows % rows_per_tile != 0) {
    throw std::invalid_argument("FilterBuffer::LoadWeightFromDram: rows_per_tile must divide the buffer capacity (4608).");
  }
  const uint32_t tiles_capacity = static_cast<uint32_t>(kFilterRows / rows_per_tile);
  if (tiles_capacity == 0) {
    throw std::logic_error("FilterBuffer::LoadWeightFromDram: tiles_capacity computed as 0.");
  }

  // Clear existing residency (we will refill from the requested tile forward).
  ClearAllOwnership();
  for (auto& r : rows_) r.fill(0); // optional but keeps debugging clean

  // Byte math per tile and per-transaction timing
  const uint32_t bytes_per_tile = static_cast<uint32_t>(rows_per_tile) * kNumPE * sizeof(std::int8_t);
  const uint32_t bw              = std::max(1u, wtiming_.bw_bytes_per_cycle);
  const uint64_t data_cycles     = CeilDivU64(static_cast<uint64_t>(bytes_per_tile),
                                              static_cast<uint64_t>(bw));
  const uint64_t xact_overhead   = static_cast<uint64_t>(wtiming_.fixed_latency);

  // How many tiles to load this time (fill as much as possible)
  const uint32_t tiles_to_load = std::min<uint32_t>(tiles_capacity, total_tiles);

  uint32_t total_bytes_loaded = 0;
  uint32_t base_row           = 0;

  for (uint32_t i = 0; i < tiles_to_load; ++i) {
    const uint32_t cur_id = (tile_id + i) % total_tiles;

    // Destination pointer starts at rows_[base_row]
    void* dst = static_cast<void*>(const_cast<std::int8_t*>(rows_[base_row].data()));
    const uint32_t n = dram_->LoadWeightTile(layer_id, cur_id, dst, bytes_per_tile);
    // Record residency and base row mapping
    owned_tile_id_.insert(cur_id);
    tile_base_row_[cur_id] = base_row;

    // Set the first one as active
    if (i == 0) active_tile_id_ = cur_id;

    // Timing accumulation (one transaction per tile)
    if (out_cycles) *out_cycles += (data_cycles + xact_overhead);
    total_bytes_loaded += n;

    base_row += static_cast<uint32_t>(rows_per_tile);
    if (base_row >= kFilterRows) break; // safety guard; should match tiles_to_load anyway
  }

  return total_bytes_loaded;
}
}
