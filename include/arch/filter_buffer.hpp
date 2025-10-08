#pragma once
// All comments are in English.

#include <array>
#include <cstdint>
#include <stdexcept>

#include "common/constants.hpp"
#include "arch/dram/simple_dram.hpp"
namespace sf { namespace dram {
  // Forward-declare SimpleDRAM to avoid forcing include path here.
  class SimpleDRAM;
}} // namespace sf::dram

namespace sf {

/**
 * FilterBuffer (Plan C, members-only ComputeRowId)
 *
 * rows[c_in][r][c][0..127] is flattened as ((c_in * K_h) + r) * K_w + c.
 * DRAM layout: [tile][input_channel][kh][kw][0..127]
 *
 * - Configure(...) sets layer-wise static parameters and DRAM ptr.
 * - Update(h_out, w_out) sets current output spatial coords.
 * - ComputeRowId(neuron_id) uses ONLY members (Cin, Win, Sh, Sw, Ph, Pw, Kh, Kw, h_out_cur, w_out_cur).
 */
class FilterBuffer {
public:
  using Row = std::array<std::int8_t, kNumPE>;

  FilterBuffer() = default;

  // Layer-wise configuration (static).
  void Configure(int C_in, int W_in,
                 int Kh, int Kw,
                 int Sh, int Sw,
                 int Ph, int Pw,
                 sf::dram::SimpleDRAM* dram_ptr);

  // Per-step update of the current output site (h_out, w_out).
  void Update(int h_out, int w_out);

  // Compute row id using ONLY member configuration/state.
  // Returns -1 if the tap maps outside the kernel window (padding/invalid).
  int ComputeRowId(std::uint32_t neuron_id) const;

  // Return a row by id (by value).
  Row GetRow(int row_id) const;

  // Load a weight tile for the current layer/tile into rows[].
  std::uint32_t LoadWeightFromDram(std::uint32_t layer_id, std::uint32_t tile_id);

  // Optional helper.
  std::size_t NumRows() const { return kFilterRows; }

private:
  // Fixed-capacity storage: 4068 rows × 128 weights
  std::array<Row, kFilterRows> rows_{};

  // Layer-wise configuration
  int C_in_ = 0;   // input channels
  int W_in_ = 0;   // input width
  int K_h_  = 0;   // kernel height
  int K_w_  = 0;   // kernel width
  int S_h_  = 0;   // stride (height)
  int S_w_  = 0;   // stride (width)
  int P_h_  = 0;   // padding (height)
  int P_w_  = 0;   // padding (width)

  // Per-step state (current output site)
  int h_out_cur_ = 0;
  int w_out_cur_ = 0;

  // DRAM interface (non-owning)
  sf::dram::SimpleDRAM* dram_ = nullptr;
};

} // namespace sf
