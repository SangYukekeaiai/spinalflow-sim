// All comments are in English.
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <cctype>
#include <string>
#include <vector>

#include "runner/simulation.hpp"

namespace {

sf::LayerSpec SelectLayer(const std::vector<sf::LayerSpec>& specs, int target_L) {
  for (const auto& spec : specs) {
    if (spec.L == target_L) {
      return spec;
    }
  }
  throw std::runtime_error("Config does not contain layer L=" + std::to_string(target_L));
}

void PrintUsage(const char* argv0) {
  std::cerr << "Usage: " << argv0 << " <dram_image.bin> <config.json> [--stats=on|off|true|false|1|0|--no-stats] [--reuse-csv=on|off|true|false|1|0|--no-reuse-csv] [--setuniq-csv=on|off|true|false|1|0|--no-setuniq-csv] [--trace=on|off|true|false|1|0|--no-trace]\n"
            << "Runs only layer L=5 from <config.json> across a fixed cache sweep.\n"
            << "  --stats=on (default) writes CSV summaries\n"
            << "  --stats=off or --no-stats disables all CSVs\n"
            << "  --reuse-csv=* toggles reuse-distance distribution CSVs\n"
            << "  --setuniq-csv=* toggles per-set unique-address CSVs\n"
            << "  --trace=* enables/disables cache trace files\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    PrintUsage(argv[0]);
    return 1;
  }

  const std::string bin_path = argv[1];
  const std::string json_path = argv[2];
  bool write_stats_csv = true;
  bool write_reuse_csv = false;
  bool write_setuniq_csv = true;
  bool reuse_csv_overridden = false;
  bool setuniq_csv_overridden = false;
  bool write_cache_traces = true;
  for (int i = 3; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") { PrintUsage(argv[0]); return 0; }
    if (arg == "--no-reuse-csv") { write_reuse_csv = false; reuse_csv_overridden = true; continue; }
    if (arg == "--no-setuniq-csv") { write_setuniq_csv = false; setuniq_csv_overridden = true; continue; }
    if (arg == "--no-trace") { write_cache_traces = false; continue; }
    if (arg == "--no-stats") { write_stats_csv = false; if (!reuse_csv_overridden) write_reuse_csv = false; if (!setuniq_csv_overridden) write_setuniq_csv = false; continue; }
    const std::string ks = "--stats=";
    if (arg.rfind(ks, 0) == 0) {
      std::string v = arg.substr(ks.size());
      for (auto& c : v) c = static_cast<char>(::tolower(c));
      if (v == "off" || v == "false" || v == "0") write_stats_csv = false;
      else if (v == "on" || v == "true" || v == "1") write_stats_csv = true;
      else { std::cerr << "Unknown value for --stats: " << v << '\n'; return 2; }
      if (!reuse_csv_overridden) write_reuse_csv = write_stats_csv;
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
    auto specs = sf::ParseConfig(json_path);
    if (specs.empty()) {
      throw std::runtime_error("Config contains no layers.");
    }

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

    auto dram = sf::InitDram(bin_path, json_path);

    std::vector<sf::LayerSpec> layer_specs;
    layer_specs.push_back(SelectLayer(specs, 5));
    layer_specs.push_back(SelectLayer(specs, 6));

    const std::vector<std::size_t> cache_sizes_bytes = {
        // 72u * 1024u,
        144u * 1024u,
        // 288u * 1024u,
        // 576u * 1024u
    };
    const std::vector<int> cache_way_options = {4};
    const std::vector<int> prefetch_depth_options = {0};
    const std::vector<sf::arch::cache::EvictionPolicy> policies = {
        sf::arch::cache::EvictionPolicy::kScoreboard,
        // sf::arch::cache::EvictionPolicy::kLRU
    };

    sf::RunNetworkWithCacheOptions(layer_specs,
                                   &dram,
                                   repo_name,
                                   model_name,
                                   cache_sizes_bytes,
                                   cache_way_options,
                                   prefetch_depth_options,
                                   policies,
                                   write_stats_csv,
                                   write_reuse_csv,
                                   /*write_scoreboard_csv=*/write_stats_csv,
                                   /*write_visit_count_distribution_csv=*/false,
                                   write_setuniq_csv,
                                   write_cache_traces,
                                   /*write_ts_duration_csv=*/false);

    std::cout << "[Simulation][Test] Completed layer-5 cache sweep successfully.\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "[Simulation][Test] Error: " << ex.what() << "\n";
    return 2;
  }
}
