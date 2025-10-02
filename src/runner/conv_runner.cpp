// runner/conv_runner.cpp
#include "runner/conv_runner.hpp"

namespace sf {

void ConvRunner::BindFetcherForLayer(int L) {
  using namespace sf::dram;
  fetch_holder_ = std::make_shared<InputSpineFetcher>(*spine_fmt_, *img_, *dir_, static_cast<std::uint16_t>(L));
  core_.SetDramFetcher(
    [fh = fetch_holder_](int batch, int spine, std::vector<std::uint8_t>& line, const sf::dram::DramFormat*& out_fmt) {
      (void)batch;
      return (*fh)(batch, spine, line, out_fmt);
    }
  );
}

bool ConvRunner::LoadOCGroup(int ocg) {
  using namespace sf::dram;
  core_.filter_buffer().SetConvShape(spec_.shape);
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
        return true;
      }
    }
  }
  return false; // safety cap tripped
}

bool ConvRunner::RunOneLayer() {
  using namespace sf::dram;

  // 0) LUT per shape
  core_.lut().SetFromConvShape(spec_.shape);
  core_.lut().Build();

  // 1) Bind input (PSB) fetcher for this layer L
  BindFetcherForLayer(spec_.L);

  // 2) Create writer for L.outputs (SEG_OUTPUT, v2 header inside)
  writer_holder_ = std::make_shared<OutputSpineWriter>(
      *spine_fmt_, *dram_bytes_, static_cast<std::uint16_t>(spec_.L) /* layer_id for SEG_OUTPUT */);

  // 3) For each OC tile: load weights; for each (h,w) position:
  //    set active output spine; run until idle; at tile end, flush + drain and bulk write.
  if (!spine_of_hw_) {
    spine_of_hw_ = [W = spec_.W](int h, int w) { return DefaultSpineOfHW(h, w, W); };
  }

  for (int tile = 0; tile < spec_.OC_tiles; ++tile) {
    (void)LoadOCGroup(tile);   // ok if sparse
    // Iterate over all positions for this tile
    for (int h = 0; h < spec_.H; ++h) {
      for (int w = 0; w < spec_.W; ++w) {
        const std::uint16_t spine_id = spine_of_hw_(h, w);
        core_.SetActiveOutputSpine(spine_id);

        // Run pipeline until it becomes idle for this position.
        const bool ok_quiesce = ClockUntilIdle();
        if (!ok_quiesce) return false;
      }
    }

    // Tile boundary: flush all partial lines from OutputQueue.
    core_.output_queue().FlushAllPartialLines();

    // Drain ready lines and bulk write them to DRAM with row_id = tile.
    std::vector<LinePacket> packets;
    packets.reserve(static_cast<std::size_t>(spec_.H * spec_.W)); // heuristic
    core_.output_queue().DrainAllReadyLines(packets);

    // Writer encodes bytes, sets SegmentHeader v2 (kind=SEG_OUTPUT), and uses row_id=tile.
    writer_holder_->BulkWriteLines(static_cast<std::uint16_t>(tile), packets);

    // Optionally run a few cycles to let downstream bookkeeping progress (not strictly needed).
    (void)ClockUntilIdle();
  }

  // 4) End-of-layer safety: ensure nothing left (should be empty).
  core_.output_queue().FlushAllPartialLines();
  std::vector<LinePacket> tail_packets;
  core_.output_queue().DrainAllReadyLines(tail_packets);
  if (!tail_packets.empty()) {
    // In well-formed flows this should be empty at this point, but we accept a final bulk write.
    writer_holder_->BulkWriteLines(static_cast<std::uint16_t>(spec_.OC_tiles - 1), tail_packets);
  }

  // 5) Finalize and record outputs range for this layer L.
  const Range out_range = writer_holder_->Finalize();
  last_out_range_ = out_range;
  if (opt_.register_layer_outputs) {
    dir_->set_output_range(spec_.L, out_range);
  }
  return true;
}

} // namespace sf
