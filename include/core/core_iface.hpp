#pragma once
// All comments are in English.
#include "common/entry.hpp"

namespace sf {

/**
 * Minimal core-facing interface for draining components.
 * Components call this to deliver entries to the final output sink
 * (e.g., a DRAM writer) without knowing builder/core internals.
 */
class CoreIface {
public:
    virtual ~CoreIface() = default;

    // Return true if the sink accepted the entry and the component should pop it.
    virtual bool SendToOutputSink(const Entry& e) = 0;
};

} // namespace sf
