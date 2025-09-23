#pragma once
#include <cstdint>
#include <string>
#include "common/constants.hpp"

namespace sf { namespace driver {

/**
 * BatchSpineMap (minimal)
 * Only records how many batches are required by the layer/input.
 * No address tables; the driver/reader is responsible for streaming data.
 */
class BatchSpineMap {
public:
  explicit BatchSpineMap(int numBatches = 1) : num_batches_(numBatches) {}

  void SetNumBatches(int n) {
    if (n <= 0 || n > kMaxBatches) throw std::out_of_range("numBatches");
    num_batches_ = n;
  }

  int NumBatches() const noexcept { return num_batches_; }

  std::string DebugString() const {
    return "BatchSpineMap{num_batches=" + std::to_string(num_batches_) + "}";
  }

private:
  int num_batches_{1};
};

}} // namespace sf::driver
