// All comments are in English.
#pragma once

#include <filesystem>
#include <fstream>
#include <string>

#include "layer_stats_summary.h"

namespace test::stats {

// Writes a CSV header if the file does not exist yet.
inline void EnsureCsvHasHeader(const std::filesystem::path& path) {
  if (std::filesystem::exists(path)) {
    return;
  }
  std::filesystem::create_directories(path.parent_path());
  std::ofstream ofs(path, std::ios::out | std::ios::trunc);
  ofs << "layer,demand_accesses,hits,misses,hit_cycles,miss_cycles,"
         "total_cycles,hit_rate,unique_demand_lines\n";
}

// Appends one summary row to the CSV file.
inline void AppendRow(const std::filesystem::path& path,
                      const LayerStatsSummary& summary) {
  std::ofstream ofs(path, std::ios::out | std::ios::app);
  ofs << summary.layer << ','
      << summary.demand_accesses << ','
      << summary.hits << ','
      << summary.misses << ','
      << summary.hit_cycles << ','
      << summary.miss_cycles << ','
      << summary.total_cycles << ','
      << summary.hit_rate << ','
      << summary.unique_demand_lines << '\n';
}

} // namespace test::stats
