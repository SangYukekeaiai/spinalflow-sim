#include "arch/filter_buffer.hpp"
#include <cstring>

namespace sf {

FilterBuffer::FilterBuffer() {
    banks_.resize(kNumBanks);
    for (int b = 0; b < kNumBanks; ++b) {
        banks_[b].resize(kRowsPerBank);
    }
}

void FilterBuffer::LoadFromDRAM(const int8_t* base, std::size_t bytes) {
    if (!base) throw std::invalid_argument("LoadFromDRAM: null base pointer");
    if (bytes > kTotalBytes) throw std::length_error("LoadFromDRAM: bytes exceed capacity");
    if ((bytes % kRowBytes) != 0) throw std::invalid_argument("LoadFromDRAM: size not multiple of row width");

    const std::size_t rows_to_load = bytes / kRowBytes;
    if (rows_to_load > static_cast<std::size_t>(kTotalRows)) {
        throw std::length_error("LoadFromDRAM: rows exceed capacity");
    }

    // Interleaved by row across banks
    for (std::size_t r = 0; r < rows_to_load; ++r) {
        int bank = 0, row_in_bank = 0;
        MapRowInterleaved(static_cast<int>(r), bank, row_in_bank);
        std::memcpy(banks_[bank][row_in_bank].data(),
                    base + r * kRowBytes,
                    kRowBytes);
    }

    // Invalidate cache
    cached_row_id_ = -1;
}

bool FilterBuffer::ReadRow(int row_id, Row& out) const {
    if (row_id < 0 || row_id >= kTotalRows) return false;

    if (cached_row_id_ == row_id) {
        out = cached_row_;
        return true;
    }

    int bank = 0, row_in_bank = 0;
    MapRowInterleaved(row_id, bank, row_in_bank);
    out = banks_[bank][row_in_bank];

    // Update cache
    cached_row_id_ = row_id;
    cached_row_    = out;
    return true;
}

std::vector<int8_t> FilterBuffer::DispatchToPEs(const Entry& entry,
                                                std::vector<PE>& pes,
                                                int8_t threshold) {
    std::vector<int8_t> outs;
    outs.reserve(pes.size());

    // In this demo we simply use neuron_id as global row id.
    const int row_id = static_cast<int>(entry.neuron_id);
    Row row{};
    const bool ok = ReadRow(row_id, row);
    if (!ok) {
        // If out-of-range, feed zeros to keep the model running.
        for (std::size_t i = 0; i < pes.size(); ++i) {
            int8_t out = pes[i].Process(static_cast<int8_t>(entry.ts), /*weight=*/0, threshold);
            outs.push_back(out);
        }
        return outs;
    }

    // Broadcast: each PE consumes its corresponding byte in the row.
    const std::size_t N = pes.size();
    for (std::size_t i = 0; i < N; ++i) {
        // If there are more PEs than row width (shouldn't happen here), wrap around.
        const int8_t w = row[i % kRowBytes];
        const int8_t out = pes[i].Process(static_cast<int8_t>(entry.ts), w, threshold);
        outs.push_back(out);
    }
    return outs;
}

void FilterBuffer::SetConvShape(const sf::dram::ConvShape& s) {
    // Validate capacity: we need IC*KH*KW rows for the currently loaded oc_group.
    const std::size_t rows_needed =
        static_cast<std::size_t>(s.IC) * s.KH * s.KW;
    if (rows_needed == 0) {
        throw std::invalid_argument("SetConvShape: rows_needed is zero (invalid shape)");
    }
    if (rows_needed > static_cast<std::size_t>(kTotalRows)) {
        throw std::length_error("SetConvShape: rows_needed exceeds FilterBuffer capacity");
    }
    shape_ = s;
    // Invalidate row cache since addressing semantics changed.
    cached_row_id_ = -1;
}

std::vector<int8_t> FilterBuffer::DispatchToPEs_KKIC(
    uint8_t ky, uint8_t kx, uint16_t in_c,
    const Entry& entry, std::vector<PE>& pes, int8_t threshold)
{
    std::vector<int8_t> outs;
    outs.reserve(pes.size());

    // MVP: FilterBuffer holds exactly one oc_group at a time.
    // Compute the "local" row id within that oc_group.
    const int row_id = RowId_KKIC_Local(ky, kx, in_c);

    Row row{};
    const bool ok = (row_id >= 0) && ReadRow(row_id, row);
    if (!ok) {
        // If out-of-range, feed zeros to keep the model running.
        for (std::size_t i = 0; i < pes.size(); ++i) {
            const int8_t out = pes[i].Process(static_cast<int8_t>(entry.ts), /*weight=*/0, threshold);
            outs.push_back(out);
        }
        return outs;
    }

    // Broadcast one byte per PE.
    const std::size_t N = pes.size();
    for (std::size_t i = 0; i < N; ++i) {
        const int8_t w = row[i % kRowBytes];
        const int8_t out = pes[i].Process(static_cast<int8_t>(entry.ts), w, threshold);
        outs.push_back(out);
    }
    return outs;
}

} // namespace sf
