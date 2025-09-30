#pragma once
#include <cstdint>

namespace sf { namespace dram {

/**
 * ConvShape
 * Minimal static shape for one layer in the OC-group-at-a-time model.
 * - KH, KW: kernel spatial size
 * - IC:     total input channels
 *
 * NOTE: FilterBuffer holds exactly one oc_group at a time in the MVP.
 * If you later store multiple oc_groups simultaneously, extend the
 * row-id mapping accordingly.
 */
struct ConvShape {
  std::uint16_t KH = 1;
  std::uint16_t KW = 1;
  std::uint16_t IC = 1;
};

}} // namespace sf::dram
