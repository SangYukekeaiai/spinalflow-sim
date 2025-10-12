// All comments are in English.
#include <filesystem>
#include <string>
#include <vector>
#include <exception>
#include <iostream>

#include "runner/simulation.hpp"

int main(int argc, char** argv) {
  // std::cout << "Entry size is " << sizeof(sf::Entry) << " bytes\n";
  // Usage: ./sim <bin_path> <json_path>
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <dram_image.bin> <config.json>\n";
    return 1;
  }

  const std::string bin_path  = argv[1];
  const std::string json_path = argv[2];

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

    // (3) Run all layers in order
    sf::RunNetwork(specs, &dram, repo_name, model_name);

    std::cout << "[Simulation] Completed successfully.\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "[Simulation] Error: " << ex.what() << "\n";
    return 2;
  }
}
