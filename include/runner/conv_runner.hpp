#pragma once
#include <cstdint>
#include <functional>
#include <vector>
#include <string>
#include <stdexcept>
#include <memory>

#include "core/clock.hpp"                  // ClockCore
#include "arch/dram/input_spine_fetcher.hpp"
#include "arch/dram/output_spine_writer.hpp" // NEW: writer for outputs
#include "arch/dram/weight_loader.hpp"     // WeightLoader
#include "arch/dram/conv_shape.hpp"        // ConvShape
#include "arch/dram/stream_reader.hpp"     // DramImage, LayerDirectory
#include "arch/dram/dram_format.hpp"       // DramFormat
#include "arch/dram/layer_directory.hpp"   // Range

namespace sf {

/**
 * ConvLayerSpec
 * Minimal layer description required by the runner.
 * - L:          layer index
 * - H, W:       number of spatial spines (assumed logical_spine_id = h*W + w)
 * - OC_tiles:   number of output-channel tiles (OC / 128)
 * - shape:      kernel shape (KH, KW, IC)
 *
 * NOTE:
 *   This runner assumes logical_spine_id == h*W + w for the layer.
 *   If your physical PSB lane indices differ from logical ids, ensure
 *   your InputSpineFetcher maps lanes -> logical ids (or make them equal).
 */
struct ConvLayerSpec {
  int                 L        = 0;
  int                 H        = 1;
  int                 W        = 1;
  int                 OC_tiles = 1;        // OC / 128
  sf::dram::ConvShape shape    {};         // KH, KW, IC
};

/**
 * ConvRunner
 * Drives one layer using:
 *  - InputSpineFetcher (PSB S5 source)
 *  - WeightLoader (FilterBuffer weights)
 *  - OutputSpineWriter (collect outputs as next-layer inputs)
 *  - ClockCore pipeline (S0..S5)
 *
 * Usage:
 *   ConvRunner runner(core, spineFmt, weightFmt, img, dram_bytes, dir, spec);
 *   runner.SetSpineOfEntryFn(...);   // REQUIRED: map Entry -> logical_spine_id
 *   runner.RunOneLayer();            // blocking until layer completes
 *   // Now dir.input_range(L+1) has been set to written [begin,end)
 */
class ConvRunner {
public:
  // Entry -> logical spine id (e.g., h*W + w). Must be provided by the user.
  using SpineIdFn = std::function<std::uint16_t(const Entry&)>;

  struct Options {
    // Clock until the pipeline makes no progress for 'idle_limit' consecutive calls.
    int idle_limit      = 256;
    // Global safety cap on total cycles to avoid infinite loops in early bring-up.
    long long cycle_cap = 10'000'000;
    // Whether to register outputs(L) as inputs(L+1) in the directory.
    bool register_next_layer_inputs = true;
  };

public:
  ConvRunner(ClockCore&                      core,
             const sf::dram::DramFormat&     spine_fmt,
             const sf::dram::DramFormat&     weight_fmt,
             const sf::dram::DramImage&      img_ro,
             std::vector<std::uint8_t>&      dram_bytes_rw, // mutable DRAM image
             sf::dram::LayerDirectory&       dir_rw,         // mutable directory
             ConvLayerSpec                    spec,
             Options                          opt = {})
  : core_(core),
    spine_fmt_(&spine_fmt),
    weight_fmt_(&weight_fmt),
    img_(&img_ro),
    dram_bytes_(&dram_bytes_rw),
    dir_(&dir_rw),
    spec_(spec),
    opt_(opt)
  {}

  // REQUIRED: provide an output spine mapper before RunOneLayer.
  void SetSpineOfEntryFn(SpineIdFn fn) { spine_of_entry_ = std::move(fn); }

  // Optional: also allow direct output sink override (will supersede writer).
  void SetOutputSink(std::function<bool(const Entry&)> sink) { external_sink_ = std::move(sink); }

  // Run the layer to completion (OC_tile by OC_tile).
  // Returns true on success (no early termination by cycle cap).
  bool RunOneLayer();

  // After RunOneLayer, query the written range (valid after writer finalize).
  sf::dram::Range LastOutputRange() const { return last_out_range_; }

private:
  // Wire the PSB fetcher for this layer (spine segments).
  void BindFetcherForLayer(int L);

  // Bind the output writer as the core's sink (unless external_sink_ is set).
  void BindWriterForNextLayer(int next_L);

  // Load an oc_group of weights into FilterBuffer and validate shape capacity.
  bool LoadOCGroup(int ocg);

  // Clock core until it is idle for 'idle_limit' consecutive ticks or cycle cap trips.
  bool ClockUntilIdle();

private:
  ClockCore&                      core_;
  const sf::dram::DramFormat*     spine_fmt_;
  const sf::dram::DramFormat*     weight_fmt_;
  const sf::dram::DramImage*      img_;         // read-only view
  std::vector<std::uint8_t>*      dram_bytes_;  // mutable backing store (for writer)
  sf::dram::LayerDirectory*       dir_;         // mutable directory
  ConvLayerSpec                   spec_;
  Options                         opt_;

  SpineIdFn                       spine_of_entry_{};
  std::function<bool(const Entry&)> external_sink_{};

  // Keep fetcher/writer alive across lambdas.
  std::shared_ptr<sf::dram::InputSpineFetcher>  fetch_holder_;
  std::shared_ptr<sf::dram::OutputSpineWriter>  writer_holder_;

  // Recorded after Finalize().
  sf::dram::Range last_out_range_{};
};

} // namespace sf
