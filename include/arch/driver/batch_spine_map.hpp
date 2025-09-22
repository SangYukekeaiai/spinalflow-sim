#pragma once
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// Use global constants
#include "common/constants.hpp"

namespace sf {
namespace driver {

class BatchSpineMap {
public:
  using Addr = std::uint64_t;

  struct Batch {
    std::array<std::vector<Addr>, kNumSpines> phys_to_logic;
  };

  explicit BatchSpineMap(int numBatches = 1);

  void SetNumBatches(int n);
  int  NumBatches() const noexcept { return num_batches_; }

  void Add(int batchIdx, int physSpineId, Addr dramAddr);
  void Set(int batchIdx, int physSpineId, const std::vector<Addr>& addrs);
  const std::vector<Addr>& Get(int batchIdx, int physSpineId) const;
  std::vector<Addr>&       Mutable(int batchIdx, int physSpineId);

  void ClearBatch(int batchIdx);
  void ClearAll();

  std::string DebugString() const;

private:
  int num_batches_{1};
  std::array<Batch, kMaxBatches> batches_{};

  void CheckBatch(int batchIdx) const;
  void CheckSpine(int physSpineId) const;
};

} // namespace driver
} // namespace sf
