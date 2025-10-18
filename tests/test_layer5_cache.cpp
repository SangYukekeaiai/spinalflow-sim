// All comments are in English.
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
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
  std::cerr << "Usage: " << argv0 << " <dram_image.bin> <config.json>\n"
            << "Runs only layer L=5 from <config.json> across a fixed cache sweep.\n";
}

}  // namespace

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

    std::vector<sf::LayerSpec> layer_specs;
    layer_specs.push_back(SelectLayer(specs, 5));

    const std::vector<std::size_t> cache_sizes_bytes = {
        72u * 1024u,
        144u * 1024u,
        288u * 1024u,
        576u * 1024u
    };
    const std::vector<int> cache_way_options = {4, 8, 32};
    const std::vector<int> prefetch_depth_options = {4};
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
                                   policies);

    std::cout << "[Simulation][Test] Completed layer-5 cache sweep successfully.\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "[Simulation][Test] Error: " << ex.what() << "\n";
    return 2;
  }
}
