// simple_dram.hpp
// All comments are in English.
#pragma once
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <cstring>
#include <stdexcept>
#include <string>      // NEW
#include <utility>     // NEW

namespace sf { namespace dram {

struct SpineMeta {
  uint32_t id = 0;        // logical spine id
  uint64_t addr = 0;      // byte address in DRAM space
  uint32_t size = 0;      // bytes
};

struct WeightTileMeta {
  uint32_t tile = 0;      // tile id
  uint64_t addr = 0;      // byte address in DRAM space
  uint32_t size = 0;      // bytes
};

struct LayerMeta {
  std::unordered_map<uint32_t, SpineMeta>        input_spines;
  std::unordered_map<uint32_t, WeightTileMeta>   weight_tiles;
  uint64_t output_write_ptr = 0;
  uint64_t output_region_begin = 0;
  uint64_t output_region_end   = 0;  // exclusive
  std::unordered_map<uint32_t, std::vector<SpineMeta>> output_segments;
};

class SimpleDRAM {
public:
  explicit SimpleDRAM(uint64_t total_bytes)
    : mem_(total_bytes, 0) {}

  // NEW: bulk-load a raw DRAM image (no headers) into mem_[0..n-1].
  // Throws if n > capacity.
  void LoadRawImage(const void* src, uint64_t n);

  // NEW: build per-layer metadata from a JSON string (no file I/O here).
  // Overwrites existing layer metas with the same L ids.
  void BuildFromJson(const std::string& json_text);

  // NEW: convenience factory that reads files and returns a populated DRAM.
  // This performs file I/O; put implementation in the .cpp.
  static SimpleDRAM FromFiles(const std::string& bin_path, const std::string& json_path);

  // Install or replace per-layer metadata (built elsewhere).
  void SetLayerMeta(uint32_t L, LayerMeta meta) {
    if (meta.output_write_ptr < meta.output_region_begin ||
        meta.output_write_ptr > meta.output_region_end)
      throw std::invalid_argument("output write ptr out of region");
    layers_[L] = std::move(meta);
  }

  // Load an input spine by logical id: memcpy into dst.
  uint32_t LoadInputSpine(uint32_t L, uint32_t spine_id, void* dst, uint32_t max_bytes) const {
    auto itL = layers_.find(L);
    if (itL == layers_.end()) throw std::out_of_range("layer not found");
    const auto& tbl = itL->second.input_spines;
    auto it = tbl.find(spine_id);
    if (it == tbl.end()) throw std::out_of_range("input spine not found");
    const SpineMeta& m = it->second;
    const uint32_t n = (m.size <= max_bytes) ? m.size : max_bytes;
    safe_copy_out(dst, m.addr, n);
    return n;
  }

  // Load a weight tile by tile id.
  uint32_t LoadWeightTile(uint32_t L, uint32_t tile_id, void* dst, uint32_t max_bytes) const {
    auto itL = layers_.find(L);
    if (itL == layers_.end()) throw std::out_of_range("layer not found");
    const auto& tbl = itL->second.weight_tiles;
    auto it = tbl.find(tile_id);
    if (it == tbl.end()) throw std::out_of_range("weight tile not found");
    const WeightTileMeta& m = it->second;
    const uint32_t n = (m.size <= max_bytes) ? m.size : max_bytes;
    safe_copy_out(dst, m.addr, n);
    return n;
  }

  // Store one output spine (append-only).
  std::uint32_t StoreOutputSpine(uint32_t L, uint32_t spine_id, const void* src, uint32_t bytes) {
    auto itL = layers_.find(L);
    if (itL == layers_.end()) throw std::out_of_range("layer not found");
    auto& meta = itL->second;
    if (meta.output_write_ptr + bytes > meta.output_region_end)
      throw std::overflow_error("output region full");
    safe_copy_in(meta.output_write_ptr, src, bytes);
    SpineMeta seg{spine_id, meta.output_write_ptr, bytes};
    meta.output_segments[spine_id].push_back(seg);
    meta.output_write_ptr += bytes;
    return bytes;
  }

private:
  void safe_copy_out(void* dst, uint64_t addr, uint32_t n) const {
    if (addr + n > mem_.size()) throw std::out_of_range("read out of range");
    std::memcpy(dst, &mem_[static_cast<size_t>(addr)], n);
  }
  void safe_copy_in(uint64_t addr, const void* src, uint32_t n) {
    if (addr + n > mem_.size()) throw std::out_of_range("write out of range");
    std::memcpy(&mem_[static_cast<size_t>(addr)], src, n);
  }

private:
  std::vector<uint8_t> mem_; // flat DRAM space
  std::unordered_map<uint32_t, LayerMeta> layers_;
};

}} // namespace sf::dram
