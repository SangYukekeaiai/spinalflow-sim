#include "arch/driver/weight_lut.hpp"
#include <sstream>

namespace sf {
namespace driver {

void WeightLUT::Build(const Params& p) {
    if (p.inC == 0 || p.outC == 0 || p.kH == 0 || p.kW == 0) {
        throw std::invalid_argument("WeightLUT::Build: zero geometry not allowed");
    }
    if (p.peLanes != 128) {
        throw std::invalid_argument("WeightLUT::Build: peLanes must be 128 to match FilterBuffer row width");
    }

    inC_ = p.inC;
    outC_ = p.outC;
    kH_ = p.kH;
    kW_ = p.kW;
    peLanes_ = p.peLanes;

    // Number of out tiles (groups of 128 output channels)
    outTiles_ = static_cast<uint16_t>((outC_ + peLanes_ - 1) / p.peLanes);
    rowsPerTile_ = static_cast<uint32_t>(inC_) * static_cast<uint32_t>(kH_) * static_cast<uint32_t>(kW_);

    built_ = true;
}

uint32_t WeightLUT::RowId(uint8_t ky, uint8_t kx, uint16_t in_c, uint16_t out_tile) const {
    CheckBuilt();
    CheckRanges(ky, kx, in_c, out_tile);
    const uint32_t base = BaseIndex(ky, kx, in_c);
    return static_cast<uint32_t>(out_tile) * rowsPerTile_ + base;
}

uint32_t WeightLUT::NeuronId(uint8_t ky, uint8_t kx, uint16_t in_c) const {
    CheckBuilt();
    if (ky >= kH_ || kx >= kW_ || in_c >= inC_) {
        throw std::out_of_range("NeuronId: ky/kx/in_c out of range");
    }
    return BaseIndex(ky, kx, in_c);
}

uint32_t WeightLUT::RowIdFromNeuron(uint32_t neuron_id, uint16_t out_tile) const {
    CheckBuilt();
    if (out_tile >= outTiles_) throw std::out_of_range("RowIdFromNeuron: out_tile out of range");
    // neuron_id == base in [0, rowsPerTile_)
    if (neuron_id >= rowsPerTile_) throw std::out_of_range("RowIdFromNeuron: neuron_id out of range");
    return static_cast<uint32_t>(out_tile) * rowsPerTile_ + neuron_id;
}

std::string WeightLUT::ToString() const {
    std::ostringstream oss;
    if (!built_) {
        oss << "WeightLUT{UNBUILT}";
        return oss.str();
    }
    oss << "WeightLUT{inC=" << inC_
        << ", outC=" << outC_
        << ", kH=" << static_cast<int>(kH_)
        << ", kW=" << static_cast<int>(kW_)
        << ", peLanes=" << peLanes_
        << ", outTiles=" << outTiles_
        << ", rowsPerTile=" << rowsPerTile_
        << "}";
    return oss.str();
}

} // namespace driver
} // namespace sf
