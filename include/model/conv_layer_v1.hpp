#pragma once
// All comments are in English.

#include <vector>
#include <cstdint>
#include <stdexcept>

#include "common/constants.hpp"
#include "arch/filter_buffer.hpp"
#include "core/core.hpp"

namespace sf {

/**
 * conv_layer
 *
 * Orchestrates site-by-site execution:
 *  - Configure FilterBuffer with layer-wide params.
 *  - For each output site (h_out, w_out):
 *      * fb.Update(h_out, w_out)
 *      * core.SetSpineContext(...)
 *      * core.ConfigureTiles(C_out)  // total_tiles = C_out / kNumPE
 *      * batches = generate_batches(h_out, w_out)
 *      * core.BindTileBatches(&batches); core.PreloadFirstBatch();
 *      * For each tile_id: fb.LoadWeightFromDram(layer_id, tile_id);
 *            while (!core.FinishedCompute()) core.StepOnce(tile_id);
 *      * After all tiles finish: core.DrainAllTilesAndStore();
 *
 * Notes:
 * - "Tile" here means an output-spine partition (e.g., each tile covers 128 output channels),
 *   not the number of PEs.
 * - Batch-map construction follows: slide kernel window; spine_id = h_in * W_in + w_in;
 *   batches_needed = ceil(Kh * Kw / kNumPhysISB); partition the spine list accordingly.
 */
class ConvLayer {
public:
  ConvLayer(Core& core, FilterBuffer& fb)
  : core_(core), fb_(fb) {}

  // Configure once per layer. H_out and W_out are derived from standard conv formula.
  void ConfigureLayer(int layer_id,
                      int C_in, int C_out,
                      int H_in, int W_in,
                      int Kh, int Kw,
                      int Sh, int Sw,
                      int Ph, int Pw,
                      sf::dram::SimpleDRAM* dram);

  // Build batches for a given output site (h_out, w_out).
  // Returns a vector of batches; each batch is a vector of logical spine_ids.
  std::vector<std::vector<int>> generate_batches(int h_out, int w_out) const;

  // Run the whole layer over all output sites.
  void run_layer();

private:
  // --- Layer parameters ---
  int layer_id_ = 0;
  int C_in_ = 0, C_out_ = 0;
  int H_in_ = 0, W_in_ = 0;
  int H_out_ = 0, W_out_ = 0;
  int Kh_ = 0, Kw_ = 0;
  int Sh_ = 0, Sw_ = 0;
  int Ph_ = 0, Pw_ = 0;

  // Subsystems
  Core&         core_;  // compute pipeline
  FilterBuffer& fb_;    // filter buffer

private:
  // Helper: derive H_out/W_out from conv formula (valid for integer output).
  static int DeriveOutDim(int in, int pad, int kernel, int stride) {
    // (in + 2*pad - kernel) must be divisible by stride
    const int numer = in + 2 * pad - kernel;
    if (numer < 0 || numer % stride != 0) {
      throw std::invalid_argument("ConvLayer: invalid shape for output dimension derivation.");
    }
    return numer / stride + 1;
  }
};

} // namespace sf
