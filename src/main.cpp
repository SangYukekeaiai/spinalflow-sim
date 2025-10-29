// All comments are in English.
#include <cctype>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "runner/simulation.hpp"

namespace {

void PrintUsage(const char* argv0) {
  std::cerr << "Usage: " << argv0
            << " <dram_image.bin> <config.json>"
            << " [--timing-csv=on|off|true|false|1|0|--timing-csv|--no-timing-csv]\n"
            << "Runs the full network described by <config.json>. CSV emission is opt-in via --timing-csv.\n";
}

bool ParseTimingCsvFlag(const std::string& arg, bool& write_stage_csv) {
  if (arg == "--timing-csv") {
    write_stage_csv = true;
    return true;
  }
  if (arg == "--no-timing-csv") {
    write_stage_csv = false;
    return true;
  }
  constexpr std::string_view prefix{"--timing-csv="};
  if (arg.rfind(prefix.data(), 0) == 0) {
    std::string value = arg.substr(prefix.size());
    for (auto& ch : value) {
      ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    if (value == "off" || value == "false" || value == "0") {
      write_stage_csv = false;
    } else if (value == "on" || value == "true" || value == "1") {
      write_stage_csv = true;
    } else {
      std::cerr << "Unknown value for --timing-csv: " << value << "\n";
      return false;
    }
    return true;
  }
  return false;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    PrintUsage(argv[0]);
    return 1;
  }

  const std::string bin_path  = argv[1];
  const std::string json_path = argv[2];

  bool write_stage_csv = false;
  for (int i = 3; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return 0;
    }
    if (!ParseTimingCsvFlag(arg, write_stage_csv)) {
      std::cerr << "Unrecognized argument: " << arg << "\n";
      PrintUsage(argv[0]);
      return 2;
    }
  }

  try {
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

    auto dram = sf::InitDram(bin_path, json_path);
    sf::RunNetwork(specs, &dram, repo_name, model_name, write_stage_csv);

    std::cout << "[Simulation] Completed successfully.\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "[Simulation] Error: " << ex.what() << "\n";
    return 2;
  }
}
