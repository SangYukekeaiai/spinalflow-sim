// All comments are in English.
#pragma once
#include <cstdint>
#include <fstream>
#include <string>
#include <stdexcept>
#include <iomanip>
#include "stats/sim_stats.hpp"

namespace sf {

// Writes one row per layer with aggregated statistics.
class LayerSummaryCsvLogger {
public:
  explicit LayerSummaryCsvLogger(const std::string& path, bool append = true)
  : path_(path)
  {
    std::ios_base::openmode mode = std::ios::out;
    mode |= (append ? std::ios::app : std::ios::trunc);
    file_.open(path_, mode);
    if (!file_.is_open()) {
      throw std::runtime_error("LayerSummaryCsvLogger: failed to open file: " + path_);
    }
    if (!append) {
      WriteHeader_();
    } else {
      file_.seekp(0, std::ios::end);
      if (file_.tellp() == 0) {
        WriteHeader_();
      }
    }
    // Make sure floating values have a consistent format.
    file_ << std::fixed << std::setprecision(4);
  }

  // Append a single layer-wide row.
  void AppendRow(
      int layer_id,
      int H_out, int W_out,
      int total_tiles,
      const sf::LayerCycleStats& S,
      int drained_entries_total)
  {
    const int sites = H_out * W_out;
    // const double step_cycles_avg_per_site =
    //     sites > 0 ? static_cast<double>(S.step_cycles_total) / static_cast<double>(sites) : 0.0;
    // const double mean_entries_per_spine =
    //     sites > 0 ? static_cast<double>(drained_entries_total) / static_cast<double>(sites) : 0.0;
    // const uint64_t input_load_cycles_total = S.preload_input_cycles + S.load_input_cycles_in_step;

    file_
      << layer_id << ','
      // << S.step_ticks << ','
      << S.step_cycles_total << ','
      // << step_cycles_avg_per_site << ','
      // << S.step_extra_memload_cycles << ','
      << S.preload_input_cycles << ','
      // << S.load_input_cycles_in_step << ','
      // << input_load_cycles_total << ','
      << S.weight_load_cycles << ','
      << S.output_drain_cycles << ','
      << S.output_store_cycles << ','
      // << drained_entries_total << ','
      // << mean_entries_per_spine << ','
      // << S.tob_in.gated_off << ',' << S.tob_in.eligible_but_noop << ',' << S.tob_in.ran << ','
      // << S.pe.gated_off     << ',' << S.pe.eligible_but_noop     << ',' << S.pe.ran << ','
      // << S.mfb.gated_off    << ',' << S.mfb.eligible_but_noop    << ',' << S.mfb.ran << ','
      // << S.isb_ld.gated_off << ',' << S.isb_ld.eligible_but_noop << ',' << S.isb_ld.ran
      << '\n';

    file_.flush();
  }

private:
  void WriteHeader_() {
    file_ <<
      // "C_in,C_out,"
      // "H_in,W_in,"
      // "Kh, Kw,"
      // "Sh, Sw,"
      // "Ph, Pw,"
      // "H_out,W_out,"
      // "total_tiles,"
      "layer_id,step_cycles_total,preload_input_cycles,weight_load_cycle,output_drain_cycles,output_store_cycles\n";

    // file_
    //   << "layer_id,"
    //      "step_ticks,step_cycles_total,step_cycles_avg_per_site,step_extra_memload_cycles,"
    //      "preload_input_cycles,load_input_cycles_in_step,input_load_cycles_total,"
    //      "weight_load_cycles,output_drain_cycles,output_store_cycles,"
    //      "drained_entries_total,mean_entries_per_spine,"
    //      "tob_in_gated,tob_in_stall,tob_in_ran,"
    //      "pe_gated,pe_stall,pe_ran,"
    //      "mfb_gated,mfb_stall,mfb_ran,"
    //      "isb_ld_gated,isb_ld_stall,isb_ld_ran\n";
  }

  std::string path_;
  std::ofstream file_;
};

} // namespace sf
