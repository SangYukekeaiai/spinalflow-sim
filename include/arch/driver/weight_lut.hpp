#pragma once
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <tuple>
#include <string>

namespace sf {
namespace driver {

/**
 * WeightLUT
 *  - CPU-side lookup for mapping (ky, kx, in_c, out_tile) -> FilterBuffer global_row_id.
 *  - Also defines a stable neuron_id for (ky, kx, in_c), independent of out_tile.
 *
 * DRAM Layout (row-major over tiles):
 *   [out_ch / 128][input channel][kh * kw][0..127]
 * where each FilterBuffer "row" holds 128 weights for the 128 PEs (one per PE lane).
 *
 * Row indexing:
 *   rows_per_tile = inC * kH * kW
 *   base          = in_c * (kH * kW) + (ky * kW + kx)
 *   global_row_id = out_tile * rows_per_tile + base
 *
 * Neuron indexing (tile-agnostic):
 *   neuron_id = base
 */
class WeightLUT {
public:
    struct Params {
        // Convolution geometry
        uint16_t inC     = 0;   // number of input channels
        uint16_t outC    = 0;   // number of output channels
        uint8_t  kH      = 0;   // kernel height
        uint8_t  kW      = 0;   // kernel width

        // Optional: sanity constraints for lanes/tiles
        uint16_t peLanes = 128; // width of a row (must be 128 to match FilterBuffer)
    };

    WeightLUT() = default;

    // Initialize the LUT for one convolution layer.
    // Throws on invalid params (zeros, non-128 lane width, etc.).
    void Build(const Params& p);

    // Return FilterBuffer global row id for (ky, kx, in_c, out_tile).
    // Precondition: Build() has been called.
    uint32_t RowId(uint8_t ky, uint8_t kx, uint16_t in_c, uint16_t out_tile) const;

    // Return a stable neuron_id for (ky, kx, in_c), independent of out_tile.
    // This lets the driver/min_finder carry a single int key per (ky,kx,in_c).
    uint32_t NeuronId(uint8_t ky, uint8_t kx, uint16_t in_c) const;

    // Reverse: given neuron_id and out_tile, recover row id.
    uint32_t RowIdFromNeuron(uint32_t neuron_id, uint16_t out_tile) const;

    // Helpers to convert an absolute output channel to (tile, lane).
    // lane  = out_c % 128 (which column in the 128B row)
    // tile  = out_c / 128 (which row block)
    static inline uint16_t LaneOf(uint32_t out_c) { return static_cast<uint16_t>(out_c % 128u); }
    static inline uint16_t TileOf(uint32_t out_c) { return static_cast<uint16_t>(out_c / 128u); }

    // Queryors
    inline uint16_t InC() const { return inC_; }
    inline uint16_t OutC() const { return outC_; }
    inline uint8_t  KH()  const { return kH_; }
    inline uint8_t  KW()  const { return kW_; }
    inline uint16_t OutTiles() const { return outTiles_; }
    inline uint32_t RowsPerTile() const { return rowsPerTile_; }
    inline bool     Built() const { return built_; }

    // Pretty string for debugging/logging.
    std::string ToString() const;

private:
    // Geometry
    uint16_t inC_ = 0;
    uint16_t outC_ = 0;
    uint8_t  kH_ = 0;
    uint8_t  kW_ = 0;
    uint16_t peLanes_ = 128;

    // Derived
    uint16_t outTiles_ = 0;
    uint32_t rowsPerTile_ = 0;

    bool built_ = false;

    // Fast path: neuron_id = base = in_c * (kH*kW) + (ky*kW + kx)
    inline uint32_t BaseIndex(uint8_t ky, uint8_t kx, uint16_t in_c) const {
        return static_cast<uint32_t>(in_c) * static_cast<uint32_t>(kH_ * kW_)
             + static_cast<uint32_t>(ky) * static_cast<uint32_t>(kW_)
             + static_cast<uint32_t>(kx);
    }

    inline void CheckBuilt() const {
        if (!built_) throw std::logic_error("WeightLUT not built. Call Build() first.");
    }

    inline void CheckRanges(uint8_t ky, uint8_t kx, uint16_t in_c, uint16_t out_tile) const {
        if (ky >= kH_ || kx >= kW_) throw std::out_of_range("ky/kx out of range");
        if (in_c >= inC_)           throw std::out_of_range("in_c out of range");
        if (out_tile >= outTiles_)  throw std::out_of_range("out_tile out of range");
    }
};

} // namespace driver
} // namespace sf
