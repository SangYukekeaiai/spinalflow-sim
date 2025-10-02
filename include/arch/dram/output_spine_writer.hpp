#pragma once
// All comments are in English.

#include <cstdint>
#include <vector>
#include <functional>
#include <cstring>
#include "common/entry.hpp"
#include "arch/output_queue.hpp"
#include "arch/dram/dram_common.hpp"
#include "arch/dram/dram_format.hpp"
#include "arch/dram/stream_writer.hpp" // if StreamWriter is placed elsewhere, adjust include

namespace sf {

/**
 * OutputSpineWriter
 *
 * Responsible for writing OutputQueue line packets back to DRAM using dram::StreamWriter.
 * It does not own the DRAM image; instead, it uses a user-provided callback to obtain
 * a writable byte buffer (std::vector<uint8_t>&) for each (layer, batch, spine).
 */
class OutputSpineWriter {
public:
    // A callback returning a writable DRAM image for (layer, batch, spine).
    // You can bind this to your DRAM manager that organizes per-layer/per-batch/per-spine images.
    using SpineSink = std::function<std::vector<std::uint8_t>& (int layer, int batch, std::uint16_t spine)>;

    OutputSpineWriter(const sf::dram::DramFormat& fmt, SpineSink sink)
        : fmt_(fmt), sink_(std::move(sink)) {}

    // Write a single line packet as one DRAM segment.
    // If you need an explicit End-Of-Spine marker, you can set it in the header when pkt.is_full == false.
    // Adapt the header population to your actual SegmentHeader definition.
    bool operator()(int layer, int batch, const LinePacket& pkt) {
        // Obtain the destination byte image for this (layer, batch, spine).
        std::vector<std::uint8_t>& image = sink_(layer, batch, pkt.spine_id);

        // Prepare the header. The exact field names may differ in your SegmentHeader.
        // Here we conservatively set at least "size" and "spine" (rename to your actual fields if needed).
        sf::dram::SegmentHeader hdr{};
        // TODO: If your header requires extra fields (e.g., timestamp range, EOL flags),
        //       fill them here accordingly.
        hdr.size  = pkt.count;               // number of entries in this segment
        hdr.logical_spine_id = pkt.spine_id;            // logical spine id for this segment
        // Optional EOL flag if your format supports it (example):
        // hdr.eol = (pkt.is_full ? 0 : 1);

        // Payload is a raw array of Entries. We assume Entry's layout matches fmt_.entry_bytes().
        const std::size_t entryB = fmt_.entry_bytes();
        const std::size_t payload_len = static_cast<std::size_t>(pkt.count) * entryB;

        // Safety check: if Entry doesn't match the DRAM entry size, convert/repack accordingly.
        if (sizeof(Entry) != entryB) {
            // Repack to a dense byte vector in the target DRAM entry format.
            temp_.resize(payload_len);
            std::memcpy(temp_.data(), pkt.entries, std::min(temp_.size(), sizeof(pkt.entries)));
            // If conversion is actually required (endianness/bit-fields), do it here instead of memcpy.
            sf::dram::StreamWriter sw(fmt_, image);
            sw.append(hdr, temp_.data(), payload_len);
        } else {
            // Fast path: structure size matches DRAM entry width, just memcpy.
            const std::uint8_t* payload = reinterpret_cast<const std::uint8_t*>(pkt.entries);
            sf::dram::StreamWriter sw(fmt_, image);
            sw.append(hdr, payload, payload_len);
        }
        return true;
    }

private:
    const sf::dram::DramFormat& fmt_;
    SpineSink                   sink_;
    std::vector<std::uint8_t>   temp_; // used only when repacking is needed
};

} // namespace sf
