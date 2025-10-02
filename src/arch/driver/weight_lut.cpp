#include "arch/driver/weight_lut.hpp"
#include <sstream>

namespace sf { namespace driver {

void WeightLUT::Reset() {
  shape_      = {1,1,1};
  outC_       = 128;
  peLanes_    = 128;
  built_      = false;
  outTiles_   = 1;
  rowsPerTile_= 1;
}

void WeightLUT::SetFromConvShape(const sf::dram::ConvShape& s) {
  shape_ = s;
  built_ = false; // defer recompute
}

void WeightLUT::SetOutChannels(std::uint16_t outC, std::uint16_t peLanes) {
  outC_    = (outC == 0 ? 128 : outC);
  peLanes_ = (peLanes == 0 ? 128 : peLanes);
  built_   = false; // defer recompute
}

void WeightLUT::Build() {
  recompute_();
}

int WeightLUT::LocalRow(std::uint8_t ky, std::uint8_t kx, std::uint16_t in_c) const {
  ensure_built_();
  check_kkic_(ky, kx, in_c);
  return (static_cast<int>(in_c) * shape_.KH + ky) * shape_.KW + kx;
}

std::uint32_t WeightLUT::RowId(std::uint8_t ky, std::uint8_t kx, std::uint16_t in_c,
                               std::uint16_t out_tile) const {
  ensure_built_();
  check_kkic_(ky, kx, in_c);
  check_tile_(out_tile);
  const std::uint32_t base = static_cast<std::uint32_t>(LocalRow(ky, kx, in_c));
  return static_cast<std::uint32_t>(out_tile) * rowsPerTile_ + base;
}

std::uint32_t WeightLUT::NeuronId(std::uint8_t ky, std::uint8_t kx, std::uint16_t in_c) const {
  ensure_built_();
  return static_cast<std::uint32_t>(LocalRow(ky, kx, in_c));
}

std::uint32_t WeightLUT::RowIdFromNeuron(std::uint32_t neuron_id, std::uint16_t out_tile) const {
  ensure_built_();
  check_tile_(out_tile);

  if (neuron_id >= rowsPerTile_) throw std::out_of_range("neuron_id out of range");
  return static_cast<std::uint32_t>(out_tile) * rowsPerTile_ + neuron_id;
}

std::string WeightLUT::ToString() const {
  std::ostringstream oss;
  if (!built_) {
    oss << "WeightLUT{UNBUILT "
        << "KH=" << shape_.KH << ", KW=" << shape_.KW << ", IC=" << shape_.IC
        << ", OutC=" << outC_   << ", peLanes=" << peLanes_
        << "}";
    return oss.str();
  }
  oss << "WeightLUT{KH=" << shape_.KH
      << ", KW=" << shape_.KW
      << ", IC=" << shape_.IC
      << ", OutC=" << outC_
      << ", peLanes=" << peLanes_
      << ", outTiles=" << outTiles_
      << ", rowsPerTile=" << rowsPerTile_
      << "}";
  return oss.str();
}

void WeightLUT::ensure_built_() const {
  if (!built_) recompute_();
}

void WeightLUT::recompute_() const {
  // Derive rowsPerTile and outTiles from current config without mutating config members.

  // 1) rowsPerTile = IC * KH * KW (guard against zero)
  std::uint32_t rp = static_cast<std::uint32_t>(shape_.IC) * shape_.KH * shape_.KW;
  if (rp == 0) rp = 1;

  // 2) effective lane width / outC for tiling (fall back to 128 if unset)
  const std::uint16_t eff_pe   = (peLanes_ == 0 ? 128 : peLanes_);
  const std::uint16_t eff_outC = (outC_    == 0 ? 128 : outC_);

  // 3) outTiles = ceil(eff_outC / eff_pe) (guard against zero)
  std::uint16_t tiles = static_cast<std::uint16_t>((eff_outC + eff_pe - 1) / eff_pe);
  if (tiles == 0) tiles = 1;

  // 4) commit to mutable derived fields
  rowsPerTile_ = rp;
  outTiles_    = tiles;
  built_       = true;
}

}} // namespace sf::driver
