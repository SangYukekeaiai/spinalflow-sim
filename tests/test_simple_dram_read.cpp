// All comments are in English.
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

#include "arch/dram/simple_dram.hpp"

// Helper: parse little-endian uint32 from 4 bytes
static inline uint32_t load_le_u32(const uint8_t* p) {
  return static_cast<uint32_t>(p[0])
       | (static_cast<uint32_t>(p[1]) << 8)
       | (static_cast<uint32_t>(p[2]) << 16)
       | (static_cast<uint32_t>(p[3]) << 24);
}

// Read JSON file entirely into nlohmann::json
static nlohmann::json read_json_file(const std::string& path) {
  std::ifstream ifs(path);
  if (!ifs) {
    throw std::runtime_error("Failed to open JSON file: " + path);
  }
  nlohmann::json j;
  ifs >> j;
  return j;
}

static void print_usage(const char* argv0) {
  std::cerr
      << "Usage:\n"
      << "  " << argv0 << " <image.bin> <meta.json>\n\n"
      << "Description:\n"
      << "  Loads DRAM from <image.bin> and layer metadata from <meta.json>.\n"
      << "  Iterates through all layers and input_spines described in JSON,\n"
      << "  reads their bytes via SimpleDRAM API, and prints parsed Entries.\n"
      << "  Each Entry is assumed to be 5 bytes: ts:uint8 + neuron_id:uint32 (little-endian).\n";
}

int main(int argc, char** argv) {
  if (argc < 3) {
    print_usage(argv[0]);
    return 1;
  }

  const std::string bin_path  = argv[1];
  const std::string json_path = argv[2];

  try {
    // 1) Build DRAM from files using provided API
    sf::dram::SimpleDRAM dram = sf::dram::SimpleDRAM::FromFiles(bin_path, json_path);

    // 2) Parse JSON again here to iterate layers/spines (metadata is private inside SimpleDRAM)
    nlohmann::json j = read_json_file(json_path);
    if (!j.contains("layers") || !j["layers"].is_array()) {
      throw std::runtime_error("JSON missing 'layers' array.");
    }

    std::size_t total_spines = 0;
    std::size_t total_entries = 0;

    for (const auto& jl : j["layers"]) {
      if (!jl.contains("L")) {
        std::cerr << "[WARN] A layer entry has no 'L' field. Skipped.\n";
        continue;
      }
      const uint32_t L = jl["L"].get<uint32_t>();
      std::cout << "===== Layer L=" << L << " =====\n";

      if (!jl.contains("input_spines")) {
        std::cout << "(No input_spines)\n";
        continue;
      }

      const auto& isp = jl["input_spines"];  // dict: spine_id_str -> {addr, size}
      for (auto it = isp.begin(); it != isp.end(); ++it) {
        const uint32_t spine_id = static_cast<uint32_t>(std::stoul(it.key()));
        const uint64_t size_u64 = it.value().at("size").get<uint64_t>();
        if (size_u64 > UINT32_MAX) {
          std::cerr << "[WARN] spine " << spine_id << " size too large, clamped to uint32_t.\n";
        }
        const uint32_t size = static_cast<uint32_t>(size_u64);

        std::vector<uint8_t> buf(size);
        // Read bytes via SimpleDRAM API
        const uint32_t n_read = dram.LoadInputSpine(L, spine_id, buf.data(), size);

        if (n_read != size) {
          std::cerr << "[WARN] spine " << spine_id << " requested " << size
                    << " bytes, got " << n_read << " bytes.\n";
          buf.resize(n_read);
        }

        std::cout << "Spine " << spine_id << ": bytes=" << buf.size();

        if (buf.empty()) {
          std::cout << " (empty)\n";
          continue;
        }

        if (buf.size() % 8 != 0) {
          std::cout << " (NOT multiple of 8; cannot parse cleanly to Entries)\n";
          // If you want to dump raw hex, uncomment below:
          // for (size_t i = 0; i < buf.size(); ++i) {
          //   if (i % 16 == 0) std::cout << "\n  ";
          //   std::cout << std::hex << std::uppercase
          //             << static_cast<int>(buf[i] >> 4)
          //             << static_cast<int>(buf[i] & 0xF) << " ";
          // }
          // std::cout << std::dec << "\n";
          continue;
        }

        const std::size_t n_entries = buf.size() / 8;
        total_spines += 1;
        total_entries += n_entries;

        std::cout << " entries=" << n_entries << "\n";
        for (std::size_t i = 0; i < n_entries; ++i) {
          const uint8_t ts = buf[8 * i + 0];
          const uint32_t nid = load_le_u32(&buf[8 * i + 4]);
          std::cout << "  [" << i << "] ts=" << static_cast<unsigned>(ts)
                    << " nid=" << nid << "\n";
        }
      }
    }

    std::cout << "===== Summary =====\n";
    std::cout << "Total spines parsed: " << total_spines << "\n";
    std::cout << "Total entries parsed: " << total_entries << "\n";

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[ERROR] " << e.what() << "\n";
    return 2;
  }
}
