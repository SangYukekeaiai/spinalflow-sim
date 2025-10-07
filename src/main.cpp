// All comments are in English.
#include <iostream>
#include <string>
#include <vector>
#include <exception>

#include "simulation.hpp"

int main(int argc, char** argv) {
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

    // (2) Init DRAM (load bin + build per-layer metadata)
    auto dram = sf::InitDram(bin_path, json_path);

    // (3) Run all layers in order
    sf::RunNetwork(specs, &dram);

    std::cout << "[Simulation] Completed successfully.\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "[Simulation] Error: " << ex.what() << "\n";
    return 2;
  }
}
