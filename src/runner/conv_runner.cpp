#include "runner/conv_runner.hpp"

namespace sf {

void ConvRunner::BindFetcherForLayer(int L) {
  using namespace sf::dram;
  // Create and hold the fetcher so the lambda has stable ownership.
  fetch_holder_ = std::make_shared<InputSpineFetcher>(*spine_fmt_, *img_, *dir_, static_cast<std::uint16_t>(L));
  core_.SetDramFetcher(
    [fh = fetch_holder_](int batch, int spine, std::vector<std::uint8_t>& line, const sf::dram::DramFormat*& out_fmt) {
      (void)batch; // batch is intentionally ignored
      return (*fh)(batch, spine, line, out_fmt);
    }
  );
}

void ConvRunner::BindWriterForNextLayer(int next_L) {
  using namespace sf::dram;

  if (external_sink_) {
    // Respect external override (user-provided sink); do not bind writer.
    core_.SetOutputSink(external_sink_);
    writer_holder_.reset();
    return;
  }

  if (!spine_of_entry_) {
    throw std::runtime_error("ConvRunner: SpineIdFn (Entry -> logical_spine_id) is not set");
  }

  // Create writer for outputs of L -> inputs of next_L.
  writer_holder_ = std::make_shared<OutputSpineWriter>(*spine_fmt_, *dram_bytes_,
                                                       static_cast<std::uint16_t>(next_L),
                                                       spine_of_entry_);
  // Bind as sink.
  core_.SetOutputSink(writer_holder_->MakeSink());
}

bool ConvRunner::LoadOCGroup(int ocg) {
  using namespace sf::dram;
  // Ensure FilterBuffer knows the shape; it validates capacity.
  core_.filter_buffer().SetConvShape(spec_.shape);

  // Pull all weight rows for 'ocg' into FilterBuffer.
  WeightLoader wl(*weight_fmt_, *img_, *dir_, static_cast<std::uint16_t>(spec_.L), spec_.shape);
  return wl.LoadOCGroupTo(core_.filter_buffer(), static_cast<std::uint16_t>(ocg));
}

bool ConvRunner::ClockUntilIdle() {
  long long cycles = 0;
  int idle_streak  = 0;

  while (cycles < opt_.cycle_cap) {
    const bool progressed = core_.run();
    ++cycles;

    if (progressed) {
      idle_streak = 0;
    } else {
      ++idle_streak;
      if (idle_streak >= opt_.idle_limit) {
        // Consider the pipeline quiescent.
        return true;
      }
    }
  }
  // Safety cap tripped.
  return false;
}

bool ConvRunner::RunOneLayer() {
  // 1) Bind PSB fetcher for this layer
  BindFetcherForLayer(spec_.L);

  // 2) If we have a next layer, bind an output writer as sink; otherwise honor external sink or keep whatever was set.
  const int next_L = spec_.L + 1;
  const bool have_next_layer = (dir_->num_layers() > next_L);
  if (have_next_layer) {
    BindWriterForNextLayer(next_L);
  } else if (external_sink_) {
    core_.SetOutputSink(external_sink_);
  }

  // 3) For each output-channel tile (oc_group), load weights and clock until idle.
  for (int ocg = 0; ocg < spec_.OC_tiles; ++ocg) {
    const bool ok_weights = LoadOCGroup(ocg);
    if (!ok_weights) {
      // It is legal to have an empty oc_group (e.g., sparse); proceed anyway.
      // You may choose to warn/log here if desired.
    }

    const bool ok_quiesce = ClockUntilIdle();
    if (!ok_quiesce) {
      // Cycle cap tripped - early bring-up guard.
      return false;
    }
  }

  // 4) Finalize writer (if any) and register directory range for the next layer.
  if (writer_holder_) {
    const sf::dram::Range out_range = writer_holder_->Finalize();
    last_out_range_ = out_range;

    if (opt_.register_next_layer_inputs && have_next_layer) {
      dir_->set_input_range(next_L, out_range);
    }
  }

  return true;
}

} // namespace sf
