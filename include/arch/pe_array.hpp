#pragma once
// All comments are in English.

#include <array>
#include <cstdint>
#include "common/constants.hpp"   // expects kNumPEs
#include "common/entry.hpp"

namespace sf {

// Forward declarations to minimize cross-includes.
class ClockCore;
class OutputQueue;
class SmallestTsPicker;

/**
 * Processing Element (PE)
 *
 * Unchanged functional PE with:
 *  - Accumulator -> Comparator -> VmemUpdate -> OutputGenerator
 *  - Optional per-PE output neuron id override (otherwise row-level id is used)
 *  - reset_vmem register configurable by set_reset_vmem()
 */
class PE {
public:
    static constexpr int8_t kNoSpike = -1;

    PE();

    // Process: accumulator -> comparator -> Vmem update -> output generator.
    int8_t Process(int8_t timestamp, int8_t filter, int8_t threshold);

    // Getters for tests
    int8_t  vmem()       const { return V_mem_; }
    bool    spiked()     const { return spiked_; }
    int8_t  reset_vmem() const { return reset_V_mem_; }

    void set_reset_vmem(int8_t v) { reset_V_mem_ = v; }

    // Optional: bind a static out neuron id for this PE (overrides row-level id).
    void          set_output_neuron_id(std::uint32_t id) { out_neuron_id_ = id; }
    std::uint32_t output_neuron_id() const { return out_neuron_id_; }

private:
    // Micro-ops
    int8_t Accumulator(int8_t V_mem, int8_t filter) const;
    bool   Comparator(int8_t new_possible_V_mem, int8_t threshold) const;
    int8_t VmemUpdate(int8_t new_possible_V_mem, bool spiked, int8_t reset_V_mem) const;
    int8_t OutputGenerator(bool spiked, int8_t timestampRegister) const;

private:
    int8_t        V_mem_;
    bool          spiked_;
    int8_t        reset_V_mem_;
    std::uint32_t out_neuron_id_ = 0xFFFFFFFFu; // invalid by default
};

/**
 * PEArray
 *
 * Stage-2 aggregation over kNumPEs PEs.
 * Contract:
 *  - Upstream latches a "row" via LatchRow(...):
 *      * common timestamp and threshold for this row
 *      * per-PE filter weight row[pe]
 *      * common out_neuron_id for the row
 *  - On run():
 *      * If core->st1_st2_valid() == false, stall (keep the latch).
 *      * Otherwise, for each PE: Process(...). If a spike is produced,
 *        emit an Entry to Stage-1 via core->ts_picker().Stage2Write(entry).
 *      * Consume the row-latch after processing all PEs.
 */
class PEArray {
public:
    PEArray();

    // Wire the core to access inter-stage valid and Stage-1 picker.
    void RegisterCore(ClockCore* core) { core_ = core; }

    // Latch one row's inputs (common timestamp/threshold/neuron_id + per-PE weights).
    void LatchRow(int8_t timestamp,
                  const std::array<int8_t, kNumPEs>& filter_row,
                  int8_t threshold,
                  std::uint32_t out_neuron_id);

    // True if a row is latched and pending execution.
    bool has_latch() const { return row_latched_valid_; }

    // Execute one Stage-2 attempt across all PEs (if allowed by st1_st2_valid()).
    // Returns true if progress was made (processed the latch).
    bool run();

    // Accessors (e.g., tests or configuration)
    std::array<PE, kNumPEs>&       pes()       { return pes_; }
    const std::array<PE, kNumPEs>& pes() const { return pes_; }

private:
    std::array<PE, kNumPEs>  pes_;

    // Row latch (broadcast parameters for this PE-array step)
    bool          row_latched_valid_ = false;
    int8_t        latched_timestamp_ = 0;
    int8_t        latched_threshold_ = 0;
    std::uint32_t latched_out_neuron_ = 0xFFFFFFFFu;
    std::array<int8_t, kNumPEs> latched_filter_row_{};

    // Back-reference to core for handshake and Stage-1 access.
    ClockCore*    core_ = nullptr; // not owned
};

} // namespace sf
