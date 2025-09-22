#pragma once
#include <string>
#include "arch/driver/batch_spine_map.hpp"

namespace sf {
namespace model {

/**
 * Pretty-print a BatchSpineMap for verification.
 * Note: exact reverse-mapping of s0 from address requires knowing totalSpines.
 * Here we focus on counts and a few addresses for quick inspection.
 */
std::string PrintBatchMap(const driver::BatchSpineMap& m);

} // namespace model
} // namespace sf
