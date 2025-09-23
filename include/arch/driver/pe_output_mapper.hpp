#pragma once
#include <cstdint>
#include "arch/pe.hpp"
#include <cstdio>


namespace sf::driver {

struct PEOutputMapper {

    // Compute and register output neuron id for a single PE.
    static inline void ComputeAndRegister(
        sf::PE& pe,
        int outC,   // total number of output channels
        int H,      // output height
        int W,      // output width
        int tile,   // tile index (each tile holds 128 output channels)
        uint8_t pe_index, // 0..127
        int h,      // current output row
        int w       // current output column
    ) {
        // Basic range checks (keep it lightweight)
        if (outC <= 0 || H <= 0 || W <= 0 || pe_index < 0 || pe_index >= 128 ||
            h < 0 || h >= H || w < 0 || w >= W || tile < 0) {
            cout:printf("PEOutputMapper: invalid params\n");
            return;
        }

        // Compute output channel for this PE in the given tile
        const int out_channel = tile * 128 + pe_index;
        if (out_channel < 0 || out_channel >= outC) {
            // This PE does not map to a valid output channel for this tile
            cout:printf("PEOutputMapper: out_channel %d invalid for outC %d\n", out_channel, outC);
            return;
        }

        // Compute linear neuron id
        const int spatial_idx = h * W + w; // row-major spatial index
        // Use uint32 to be safe for large tensors
        const std::uint32_t neuron_id =
            static_cast<std::uint32_t>(spatial_idx) * static_cast<std::uint32_t>(outC) +
            static_cast<std::uint32_t>(out_channel);

        // Register on the PE
        pe.set_output_neuron_id(neuron_id);
    }
};

} // namespace sf::driver
