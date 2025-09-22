// tests/test_weight_lut.cpp
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>
#include "arch/driver/weight_lut.hpp"

// Simple helper macro to check exceptions
#define EXPECT_THROW(stmt)                    \
    do {                                      \
        bool threw = false;                   \
        try { (void)(stmt); }                 \
        catch (...) { threw = true; }         \
        assert(threw && "Expected exception was not thrown"); \
    } while(0)

static void Test_Build_And_Derived() {
    using sf::driver::WeightLUT;

    WeightLUT lut;
    // inC=3, outC=260, kH=3, kW=3 -> rowsPerTile = 3*3*3=27, outTiles = ceil(260/128)=3
    WeightLUT::Params p;
    p.inC = 3; p.outC = 260; p.kH = 3; p.kW = 3; p.peLanes = 128;
    lut.Build(p);

    assert(lut.Built());
    assert(lut.InC() == 3);
    assert(lut.OutC() == 260);
    assert(lut.KH()  == 3);
    assert(lut.KW()  == 3);
    assert(lut.RowsPerTile() == 27u);
    assert(lut.OutTiles() == 3u);

    std::string s = lut.ToString();
    // A very loose check just to ensure content is present
    assert(s.find("inC=3") != std::string::npos);
    assert(s.find("outTiles=3") != std::string::npos);
    assert(s.find("rowsPerTile=27") != std::string::npos);
}

static void Test_RowId_Formula() {
    using sf::driver::WeightLUT;

    WeightLUT lut;
    WeightLUT::Params p{.inC=3, .outC=260, .kH=3, .kW=3, .peLanes=128};
    lut.Build(p);

    // rowsPerTile = 27
    const uint32_t rpt = lut.RowsPerTile();

    // BaseIndex(ky,kx,in_c) = in_c*(kH*kW) + ky*kW + kx
    // For (ky=1,kx=2,in_c=1): base = 1*(3*3) + 1*3 + 2 = 9 + 3 + 2 = 14
    // RowId(ky,kx,in_c,out_tile) = out_tile*rpt + base
    // For out_tile=0 -> row_id = 0*27 + 14 = 14
    // For out_tile=2 -> row_id = 2*27 + 14 = 68
    uint32_t r0 = lut.RowId(/*ky*/1, /*kx*/2, /*in_c*/1, /*out_tile*/0);
    uint32_t r2 = lut.RowId(/*ky*/1, /*kx*/2, /*in_c*/1, /*out_tile*/2);
    assert(r0 == 14u);
    assert(r2 == 68u);

    // Another point: (ky=0,kx=0,in_c=2)
    // base = 2*(9) + 0 + 0 = 18; tile=1 -> row_id=1*27 + 18 = 45
    uint32_t r1 = lut.RowId(0, 0, 2, 1);
    assert(r1 == 45u);
}

static void Test_NeuronId_and_RowIdFromNeuron() {
    using sf::driver::WeightLUT;

    WeightLUT lut;
    WeightLUT::Params p{.inC=3, .outC=260, .kH=3, .kW=3, .peLanes=128};
    lut.Build(p);

    // neuron_id = base = in_c*(kH*kW) + ky*kW + kx
    // Use (ky=2,kx=1,in_c=0) -> base = 0*9 + 2*3 + 1 = 7
    uint32_t neuron = lut.NeuronId(2, 1, 0);
    assert(neuron == 7u);

    // RowIdFromNeuron(neuron, tile) = tile*rpt + neuron
    // rpt=27; tile=2 -> row_id=2*27 + 7 = 61
    uint32_t row = lut.RowIdFromNeuron(neuron, /*out_tile*/2);
    assert(row == 61u);

    // Cross-check equivalence with RowId(ky,kx,in_c,tile)
    uint32_t row2 = lut.RowId(2, 1, 0, 2);
    assert(row2 == row);
}

static void Test_Tile_And_Lane_Helpers() {
    using sf::driver::WeightLUT;

    // LaneOf(out_c) = out_c % 128 ; TileOf(out_c) = out_c / 128
    assert(WeightLUT::LaneOf(0)   == 0);
    assert(WeightLUT::TileOf(0)   == 0);

    assert(WeightLUT::LaneOf(127) == 127);
    assert(WeightLUT::TileOf(127) == 0);

    assert(WeightLUT::LaneOf(128) == 0);
    assert(WeightLUT::TileOf(128) == 1);

    assert(WeightLUT::LaneOf(259) == (259 % 128));
    assert(WeightLUT::TileOf(259) == (259 / 128));
}

static void Test_Exceptions() {
    using sf::driver::WeightLUT;

    // 1) Not built -> methods should throw
    {
        WeightLUT lut;
        EXPECT_THROW(lut.NeuronId(0,0,0));
        EXPECT_THROW(lut.RowId(0,0,0,0));
        EXPECT_THROW(lut.RowIdFromNeuron(0,0));
        std::string s = lut.ToString(); // allowed; returns "UNBUILT"
        (void)s;
    }

    // 2) Invalid Build params
    {
        WeightLUT lut;
        WeightLUT::Params bad;
        bad.inC = 0; bad.outC = 1; bad.kH = 1; bad.kW = 1; bad.peLanes = 128;
        EXPECT_THROW(lut.Build(bad));

        bad.inC = 1; bad.peLanes = 64; // lanes must be 128
        EXPECT_THROW(lut.Build(bad));
    }

    // 3) Out-of-range arguments after Build
    {
        WeightLUT lut;
        lut.Build({.inC=3, .outC=129, .kH=3, .kW=3, .peLanes=128}); // outTiles=2, rowsPerTile=27

        // ky/kx OOR
        EXPECT_THROW(lut.RowId(/*ky*/3, /*kx*/0, /*in_c*/0, /*tile*/0));
        EXPECT_THROW(lut.RowId(/*ky*/0, /*kx*/3, /*in_c*/0, /*tile*/0));

        // in_c OOR
        EXPECT_THROW(lut.RowId(0, 0, /*in_c*/3, 0));
        // tile OOR: outTiles = ceil(129/128)=2 -> valid tiles: 0,1 -> 2 is OOR
        EXPECT_THROW(lut.RowId(0, 0, 0, /*tile*/2));

        // NeuronId OOR
        EXPECT_THROW(lut.NeuronId(/*ky*/3, 0, 0));
        EXPECT_THROW(lut.NeuronId(0, /*kx*/3, 0));
        EXPECT_THROW(lut.NeuronId(0, 0, /*in_c*/3));

        // RowIdFromNeuron: neuron_id must be < rowsPerTile (27)
        EXPECT_THROW(lut.RowIdFromNeuron(/*neuron_id*/27, /*tile*/1));
        // RowIdFromNeuron: tile must be < outTiles (2)
        EXPECT_THROW(lut.RowIdFromNeuron(/*neuron_id*/0, /*tile*/2));
    }
}

int main() {
    Test_Build_And_Derived();
    Test_RowId_Formula();
    Test_NeuronId_and_RowIdFromNeuron();
    Test_Tile_And_Lane_Helpers();
    Test_Exceptions();

    std::cout << "[OK] WeightLUT unit tests passed.\n";
    return 0;
}
