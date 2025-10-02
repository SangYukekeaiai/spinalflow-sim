// runner/conv_runner.hpp
#pragma once
#include <cstdint>
#include <functional>
#include <vector>
#include <string>
#include <stdexcept>
#include <memory>

#include "core/clock.hpp"                  // ClockCore
#include "arch/dram/input_spine_fetcher.hpp"
#include "arch/dram/output_spine_writer.hpp" // must support BulkWriteLines(tile, packets)
#include "arch/dram/weight_loader.hpp"       // WeightLoader
#include "arch/dram/conv_shape.hpp"          // ConvShape
#include "arch/dram/stream_reader.hpp"       // DramImage, LayerDirectory
#include "arch/dram/dram_format.hpp"         // DramFormat
#include "arch/dram/layer_directory.hpp"     // Range

namespace sf {

// Runner options
struct ConvRunnerOptions {
  // All comments in English
  int       idle_limit = 256;
  long long cycle_cap  = 10'000'000;

  // Register outputs range of this layer into directory.
  bool register_layer_outputs = true;
};

// Minimal layer spec
struct ConvLayerSpec {
  int                 L        = 0;
  int                 H        = 1;
  int                 W        = 1;
  int                 OC_tiles = 1;        // OC / 128
  sf::dram::ConvShape shape    {};         // KH, KW, IC
};

class ConvRunner {
public:
  // Map (h, w) -> logical spine id. We also provide a helper below.
  using SpineOfHWFn = std::function<std::uint16_t(int h, int w)>;

public:
  ConvRunner(ClockCore&                      core,
             const sf::dram::DramFormat&     spine_fmt,
             const sf::dram::DramFormat&     weight_fmt,
             const sf::dram::DramImage&      img_ro,
             std::vector<std::uint8_t>&      dram_bytes_rw,
             sf::dram::LayerDirectory&       dir_rw,
             ConvLayerSpec                    spec,
             ConvRunnerOptions                opt = ConvRunnerOptions{})
  : core_(core),
    spine_fmt_(&spine_fmt),
    weight_fmt_(&weight_fmt),
    img_(&img_ro),
    dram_bytes_(&dram_bytes_rw),
    dir_(&dir_rw),
    spec_(spec),
    opt_(opt)
  {}

  // REQUIRED: provide (h,w) -> logical spine id. Commonly: h*W + w
  void SetSpineOfHWFn(SpineOfHWFn fn) { spine_of_hw_ = std::move(fn); }

  // Run the layer to completion with tile-major scheduling and deferred writeback.
  bool RunOneLayer();

  // After RunOneLayer, query the outputs range (valid after writer finalize).
  sf::dram::Range LastOutputRange() const { return last_out_range_; }

  // Helper: default mapper h*W + w
  static std::uint16_t DefaultSpineOfHW(int h, int w, int W) {
    return static_cast<std::uint16_t>(h * W + w);
  }

private:
  void BindFetcherForLayer(int L);
  bool LoadOCGroup(int ocg);
  bool ClockUntilIdle();

private:
  ClockCore&                      core_;
  const sf::dram::DramFormat*     spine_fmt_;
  const sf::dram::DramFormat*     weight_fmt_;
  const sf::dram::DramImage*      img_;         // read-only
  std::vector<std::uint8_t>*      dram_bytes_;  // mutable DRAM backing store
  sf::dram::LayerDirectory*       dir_;         // mutable directory
  ConvLayerSpec                   spec_;
  ConvRunnerOptions               opt_;

  SpineOfHWFn                     spine_of_hw_{};

  std::shared_ptr<sf::dram::InputSpineFetcher>  fetch_holder_;
  std::shared_ptr<sf::dram::OutputSpineWriter>  writer_holder_;

  sf::dram::Range last_out_range_{};
};

} // namespace sf
