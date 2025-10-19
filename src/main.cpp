// All comments are in English.
#include <filesystem>
#include <string>
#include <vector>
#include <exception>
#include <iostream>
#include <cctype>

#include "runner/simulation.hpp"

static void PrintUsage(const char* argv0) {
  std::cerr << "Usage: " << argv0
            << " <dram_image.bin> <config.json>"
            << " [--stats=on|off|true|false|1|0|--no-stats]"
            << " [--reuse-csv=on|off|true|false|1|0|--no-reuse-csv]"
            << " [--scoreboard-csv=on|off|true|false|1|0|--no-scoreboard-csv]"
            << " [--setuniq-csv=on|off|true|false|1|0|--no-setuniq-csv]"
            << " [--trace=on|off|true|false|1|0|--no-trace]" << '\n'
            << "Runs the full network described by <config.json>." << '\n'
            << "  --stats=on (default) writes CSV summaries" << '\n'
            << "  --stats=off or --no-stats disables all CSVs" << '\n'
            << "  --reuse-csv=* toggles reuse-distance distribution CSVs" << '\n'
            << "  --scoreboard-csv=* toggles scoreboard score CSVs (score -> channels)" << '\n'
            << "  --setuniq-csv=* toggles per-set unique-address CSVs" << '\n'
            << "  --trace=* enables/disables cache trace files" << '\n';
}

int main(int argc, char** argv) {
  // std::cout << "Entry size is " << sizeof(sf::Entry) << " bytes\n";
  // Usage: ./sim <bin_path> <json_path>
  if (argc < 3) {
    PrintUsage(argv[0]);
    return 1;
  }

  const std::string bin_path  = argv[1];
  const std::string json_path = argv[2];

  // Optional flags controlling CSV generation (default: ON)
  bool write_stats_csv = true;
  bool write_reuse_csv = true;
  bool write_scoreboard_csv = true;
  bool write_setuniq_csv = true;
  bool write_cache_traces = true;
  bool reuse_csv_overridden = false;
  bool scoreboard_csv_overridden = false;
  bool setuniq_csv_overridden = false;
  for (int i = 3; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") { PrintUsage(argv[0]); return 0; }
    if (arg == "--no-reuse-csv") { write_reuse_csv = false; reuse_csv_overridden = true; continue; }
    if (arg == "--no-scoreboard-csv") { write_scoreboard_csv = false; scoreboard_csv_overridden = true; continue; }
    if (arg == "--no-setuniq-csv") { write_setuniq_csv = false; setuniq_csv_overridden = true; continue; }
    if (arg == "--no-trace") { write_cache_traces = false; continue; }
    if (arg == "--no-stats") { write_stats_csv = false; if (!reuse_csv_overridden) write_reuse_csv = false; if (!scoreboard_csv_overridden) write_scoreboard_csv = false; if (!setuniq_csv_overridden) write_setuniq_csv = false; continue; }
    const std::string ks = "--stats=";
    if (arg.rfind(ks, 0) == 0) {
      std::string v = arg.substr(ks.size());
      for (auto& c : v) c = static_cast<char>(::tolower(c));
      if (v == "off" || v == "false" || v == "0") write_stats_csv = false;
      else if (v == "on" || v == "true" || v == "1") write_stats_csv = true;
      else { std::cerr << "Unknown value for --stats: " << v << '\n'; return 2; }
      if (!reuse_csv_overridden) write_reuse_csv = write_stats_csv;
      if (!scoreboard_csv_overridden) write_scoreboard_csv = write_stats_csv;
      if (!setuniq_csv_overridden) write_setuniq_csv = write_stats_csv;
      continue;
    }
    const std::string k = "--reuse-csv=";
    if (arg.rfind(k, 0) == 0) {
      std::string v = arg.substr(k.size());
      for (auto& c : v) c = static_cast<char>(::tolower(c));
      if (v == "off" || v == "false" || v == "0") write_reuse_csv = false;
      else if (v == "on" || v == "true" || v == "1") write_reuse_csv = true;
      else { std::cerr << "Unknown value for --reuse-csv: " << v << '\n'; return 2; }
      reuse_csv_overridden = true;
      continue;
    }
    const std::string ksbb = "--scoreboard-csv=";
    if (arg.rfind(ksbb, 0) == 0) {
      std::string v = arg.substr(ksbb.size());
      for (auto& c : v) c = static_cast<char>(::tolower(c));
      if (v == "off" || v == "false" || v == "0") write_scoreboard_csv = false;
      else if (v == "on" || v == "true" || v == "1") write_scoreboard_csv = true;
      else { std::cerr << "Unknown value for --scoreboard-csv: " << v << '\n'; return 2; }
      scoreboard_csv_overridden = true;
      continue;
    }
    const std::string ksx = "--setuniq-csv=";
    if (arg.rfind(ksx, 0) == 0) {
      std::string v = arg.substr(ksx.size());
      for (auto& c : v) c = static_cast<char>(::tolower(c));
      if (v == "off" || v == "false" || v == "0") write_setuniq_csv = false;
      else if (v == "on" || v == "true" || v == "1") write_setuniq_csv = true;
      else { std::cerr << "Unknown value for --setuniq-csv: " << v << '\n'; return 2; }
      setuniq_csv_overridden = true;
      continue;
    }
    const std::string kt = "--trace=";
    if (arg.rfind(kt, 0) == 0) {
      std::string v = arg.substr(kt.size());
      for (auto& c : v) c = static_cast<char>(::tolower(c));
      if (v == "off" || v == "false" || v == "0") write_cache_traces = false;
      else if (v == "on" || v == "true" || v == "1") write_cache_traces = true;
      else { std::cerr << "Unknown value for --trace: " << v << '\n'; return 2; }
      continue;
    }
  }

  try {
    // (1) Parse config → vector<LayerSpec>
    auto specs = sf::ParseConfig(json_path);
    namespace fs = std::filesystem;
    const fs::path json_fs = fs::absolute(fs::path(json_path));
    std::string model_name = json_fs.parent_path().filename().string();
    std::string repo_name = json_fs.parent_path().parent_path().filename().string();
    if (model_name.empty()) {
      model_name = json_fs.stem().string();
    }
    if (repo_name.empty()) {
      repo_name = fs::current_path().filename().string();
      if (repo_name.empty()) {
        repo_name = "repo";
      }
    }

    // (2) Init DRAM (load bin + build per-layer metadata)
    auto dram = sf::InitDram(bin_path, json_path);

    // (3) Run all layers in order with default cache sweeps
    const std::vector<std::size_t> default_cache_sizes_bytes = {
        72u * 1024u,
        144u * 1024u,
        288u * 1024u,
        576u * 1024u
    };
    const std::vector<int> default_cache_way_options = {4, 8, 16};
    const std::vector<int> default_prefetch_depth_options = {1, 2, 3, 4};
    const std::vector<sf::arch::cache::EvictionPolicy> default_policies = {
        sf::arch::cache::EvictionPolicy::kScoreboard,
        sf::arch::cache::EvictionPolicy::kLRU
    };

    sf::RunNetworkWithCacheOptions(specs,
                                   &dram,
                                   repo_name,
                                   model_name,
                                   default_cache_sizes_bytes,
                                   default_cache_way_options,
                                   default_prefetch_depth_options,
                                   default_policies,
                                   write_stats_csv,
                                   write_reuse_csv,
                                   write_scoreboard_csv,
                                   /*write_visit_count_distribution_csv=*/false,
                                   write_setuniq_csv,
                                   write_cache_traces);

    std::cout << "[Simulation] Completed successfully.\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "[Simulation] Error: " << ex.what() << "\n";
    return 2;
  }
}
