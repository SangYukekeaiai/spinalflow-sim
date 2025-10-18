// All comments are in English.
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>

#include "runner/simulation.hpp"

static void PrintUsage(const char* argv0) {
  std::cerr << "Usage: " << argv0 << " <dram_image.bin> <config.json>\n"
            << "Runs only the first layer from <config.json> across a fixed cache sweep.\n";
}

int main(int argc, char** argv) {
  if (argc < 3) {
    PrintUsage(argv[0]);
    return 1;
  }

  const std::string bin_path = argv[1];
  const std::string json_path = argv[2];

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

    std::vector<sf::LayerSpec> first_layer_specs;
    first_layer_specs.push_back(specs.front());

    const std::vector<std::size_t> cache_sizes_bytes = {
        72u * 1024u,
        144u * 1024u,
        288u * 1024u,
        576u * 1024u
    };
    const std::vector<int> cache_way_options = {4, 8, 32};
    const std::vector<int> prefetch_depth_options = {4};
    const std::vector<sf::arch::cache::EvictionPolicy> policies = {
        sf::arch::cache::EvictionPolicy::kScoreboard
        // sf::arch::cache::EvictionPolicy::kLRU
    };

    sf::RunNetworkWithCacheOptions(first_layer_specs,
                                   &dram,
                                   repo_name,
                                   model_name,
                                   cache_sizes_bytes,
                                   cache_way_options,
                                   prefetch_depth_options,
                                   policies);

    std::cout << "[Simulation][Test] Completed first-layer cache sweep successfully.\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "[Simulation][Test] Error: " << ex.what() << "\n";
    return 2;
  }
}
