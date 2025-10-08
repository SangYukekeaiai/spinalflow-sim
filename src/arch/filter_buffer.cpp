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
  if (row_id < 0 || row_id >= static_cast<int>(kFilterRows)) {
    throw std::out_of_range("FilterBuffer::GetRow: row_id out of range.");
  }
  // for (std::size_t i = 0; i < kNumPE; ++i) {
    
  //   if(static_cast<int>(rows_[static_cast<std::size_t>(row_id)][i]) != 0)std::cout << static_cast<int>(rows_[static_cast<std::size_t>(row_id)][i]) << " ";
  // }
  // std::cout << "\n";
  
  return rows_[static_cast<std::size_t>(row_id)];
}

std::uint32_t FilterBuffer::LoadWeightFromDram(std::uint32_t layer_id, std::uint32_t tile_id) {
  if (!dram_) {
    throw std::runtime_error("FilterBuffer::LoadWeightFromDram: DRAM pointer is null.");
  }

  // Destination pointer to the first byte of rows[][].
  void* dst = static_cast<void*>(rows_.front().data());
  const std::uint32_t max_bytes =
      static_cast<std::uint32_t>(kFilterRows * kNumPE * sizeof(std::uint8_t));

  return dram_->LoadWeightTile(layer_id, tile_id, dst, max_bytes);
}

} // namespace sf
