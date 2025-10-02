#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include "arch/dram/conv_shape.hpp"

namespace sf { namespace driver {

/**
 * WeightLUT (MVP-friendly)
 *
 * Purpose:
 *  - Map (ky, kx, in_c [, out_tile]) to FilterBuffer row id.
 *  - Provide a stable neuron_id for (ky, kx, in_c) independent of out_tile.
 *
 * Design:
 *  - Geometry comes from layer metadata: ConvShape {KH, KW, IC}.
 *  - Optional OutC informs how many out tiles exist (OutC/peLanes); if not set,
 *    we assume a single tile (outTiles_=1), which matches the MVP "one oc_group in FB".
 *  - Lazy build: any getter ensures internal derived fields are computed.
 *  - Backward-compatible methods are kept (RowId/RowIdFromNeuron).
 */
class WeightLUT {
public:
  WeightLUT() = default;

  // Reset to defaults (KH=KW=IC=1, peLanes=128, outTiles=1).
  void Reset();

  // Set geometry from layer metadata.
  void SetFromConvShape(const sf::dram::ConvShape& s);

  // Optional: declare total output channels to compute outTiles=ceil(OutC/peLanes).
  // You can ignore this for MVP (FB holds a single oc_group at a time).
  void SetOutChannels(std::uint16_t outC, std::uint16_t peLanes = 128);

  // Explicit build is optional (lazy build happens on first query).
  void Build();

  // Query build flag.
  bool IsBuilt() const { return built_; }

  // --- Core queries ---

  // Returns the "local" row id within the currently loaded oc_group.
  // local_row = (inC * KH + ky) * KW + kx
  int LocalRow(std::uint8_t ky, std::uint8_t kx, std::uint16_t in_c) const;

  // Backward-compatible: global row id for (ky,kx,in_c,out_tile).
  // In MVP (outTiles_=1) this equals LocalRow(...).
  std::uint32_t RowId(std::uint8_t ky, std::uint8_t kx, std::uint16_t in_c,
                      std::uint16_t out_tile = 0) const;

  // Stable neuron_id independent of out_tile == LocalRow.
  std::uint32_t NeuronId(std::uint8_t ky, std::uint8_t kx, std::uint16_t in_c) const;

  // Reverse mapping: given neuron_id (local row) and out_tile -> global row id.
  std::uint32_t RowIdFromNeuron(std::uint32_t neuron_id, std::uint16_t out_tile = 0) const;

  // Helpers to split absolute out channel (only meaningful if you use multi-tiles).
  static inline std::uint16_t LaneOf(std::uint32_t out_c) { return static_cast<std::uint16_t>(out_c % 128u); }
  static inline std::uint16_t TileOf(std::uint32_t out_c) { return static_cast<std::uint16_t>(out_c / 128u); }

  // Inspectors
  const sf::dram::ConvShape& shape() const { return shape_; }
  std::uint16_t OutTiles()  const { ensure_built_(); return outTiles_; }
  std::uint32_t RowsPerTile() const { ensure_built_(); return rowsPerTile_; }
  std::uint16_t OutC()      const { return outC_; }
  std::uint16_t PELanes()   const { return peLanes_; }

  std::string ToString() const;

private:
  // Geometry (from layer metadata)
  sf::dram::ConvShape shape_{1,1,1}; // KH, KW, IC

  // Output-channel info (optional)
  std::uint16_t outC_    = 128;      // default 128 so outTiles_=1 by default
  std::uint16_t peLanes_ = 128;      // row width; must be 128 to match FB

  // Derived
  mutable bool        built_       = false;
  mutable std::uint16_t outTiles_  = 1;  // ceil(outC/peLanes)
  mutable std::uint32_t rowsPerTile_= 1; // IC*KH*KW

private:
  // Recompute derived fields if not built.
  void ensure_built_() const;
  void recompute_() const;

  // Bounds check helpers
  inline void check_kkic_(std::uint8_t ky, std::uint8_t kx, std::uint16_t in_c) const {
    if (ky >= shape_.KH || kx >= shape_.KW) throw std::out_of_range("ky/kx out of range");
    if (in_c >= shape_.IC)                  throw std::out_of_range("in_c out of range");
  }
  inline void check_tile_(std::uint16_t out_tile) const {
    if (out_tile >= outTiles_) throw std::out_of_range("out_tile out of range");
  }
};

}} // namespace sf::driver
