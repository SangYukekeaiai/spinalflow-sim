#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

#include "common/entry.hpp"
#include "arch/pe.hpp"

namespace sf {

/**
 * FilterBuffer
 *  - Total size: 576 KB
 *  - Banks: 32
 *  - Row width: 128 bytes (1024-bit bus to feed 128 PEs per cycle)
 *  - Interleaved row mapping by default:
 *        bank       = global_row_id % kNumBanks
 *        rowInBank  = global_row_id / kNumBanks
 *
 * A "row" provides 128 signed 8-bit weights (one per PE).
 */
class FilterBuffer {
public:
    static constexpr std::size_t kTotalBytes     = 576u * 1024u;  // 576 KB
    static constexpr int         kNumBanks       = 32;
    static constexpr int         kRowBytes       = 128;           // 128 weights per row
    static constexpr int         kNumPEs         = 128;           // bus width = 1024 bits
    static constexpr int         kRowsPerBank    =
        static_cast<int>(kTotalBytes / (kNumBanks * kRowBytes));  // 144
    static constexpr int         kTotalRows      = kRowsPerBank * kNumBanks; // 4608

    using Row = std::array<int8_t, kRowBytes>;

public:
    FilterBuffer();

    // Load contiguous bytes from a DRAM-like memory into the buffer.
    // 'bytes' must be a multiple of kRowBytes and must not exceed kTotalBytes.
    void LoadFromDRAM(const int8_t* base, std::size_t bytes);

    // Read a whole row by global row id into 'out'.
    bool ReadRow(int row_id, Row& out) const;

    // Convenience: dispatch the row indicated by entry.neuron_id to all PEs.
    // The mapping from neuron_id -> global_row_id is 1:1 in this demo.
    // Returns PE outputs (timestamp or PE::kNoSpike).
    std::vector<int8_t> DispatchToPEs(const Entry& entry,
                                      std::vector<PE>& pes,
                                      int8_t threshold);
    // New dispatcher that uses (ky, kx, in_c, out_tile) to pick the correct 128-wide row.
    std::vector<int8_t> DispatchToPEs_KKIC_Tile(
        uint8_t ky, uint8_t kx, uint16_t in_c, uint8_t out_tile,
        const Entry& entry, std::vector<PE>& pes, int8_t threshold);


private:
    // Map a global row id to (bank, rowInBank) with interleaving.
    static inline void MapRowInterleaved(int global_row_id, int& bank, int& row_in_bank) {
        bank       = global_row_id % kNumBanks;
        row_in_bank= global_row_id / kNumBanks;
    }

private:
    // banks_[bank][rowIdx][byte]
    std::vector<std::vector<Row>> banks_;

    // Tiny one-row cache to avoid re-reading the same row repeatedly.
    mutable int cached_row_id_{-1};
    mutable Row cached_row_{};
};

} // namespace sf
