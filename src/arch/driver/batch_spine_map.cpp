#include "arch/driver/batch_spine_map.hpp"
#include <sstream>

namespace sf {
namespace driver {

BatchSpineMap::BatchSpineMap(int numBatches) { SetNumBatches(numBatches); }

void BatchSpineMap::SetNumBatches(int n) {
  // Validate range and reset storage for deterministic behavior
  if (n < 1 || n > kMaxBatches) {
    throw std::out_of_range("SetNumBatches: n must be in [1, kMaxBatches]");
  }
  // If increasing, ensure the new batches are empty
  if (n > num_batches_) {
    for (int b = num_batches_; b < n; ++b) {
      for (auto& vec : batches_[b].phys_to_logic) vec.clear();
    }
  }
  // If decreasing, proactively clear disabled batches
  if (n < num_batches_) {
    for (int b = n; b < num_batches_; ++b) {
      for (auto& vec : batches_[b].phys_to_logic) vec.clear();
    }
  }
  num_batches_ = n;
}

void BatchSpineMap::Add(int batchIdx, int physSpineId, Addr dramAddr) {
  CheckBatch(batchIdx);
  CheckSpine(physSpineId);
  batches_[batchIdx].phys_to_logic[physSpineId].push_back(dramAddr);
}

void BatchSpineMap::Set(int batchIdx, int physSpineId, const std::vector<Addr>& addrs) {
  CheckBatch(batchIdx);
  CheckSpine(physSpineId);
  batches_[batchIdx].phys_to_logic[physSpineId] = addrs;
}

const std::vector<BatchSpineMap::Addr>&
BatchSpineMap::Get(int batchIdx, int physSpineId) const {
  CheckBatch(batchIdx);
  CheckSpine(physSpineId);
  return batches_[batchIdx].phys_to_logic[physSpineId];
}

std::vector<BatchSpineMap::Addr>&
BatchSpineMap::Mutable(int batchIdx, int physSpineId) {
  CheckBatch(batchIdx);
  CheckSpine(physSpineId);
  return batches_[batchIdx].phys_to_logic[physSpineId];
}

void BatchSpineMap::ClearBatch(int batchIdx) {
  CheckBatch(batchIdx);
  for (auto& vec : batches_[batchIdx].phys_to_logic) vec.clear();
}

void BatchSpineMap::ClearAll() {
  for (int b = 0; b < kMaxBatches; ++b) {
    for (auto& vec : batches_[b].phys_to_logic) vec.clear();
  }
}

std::string BatchSpineMap::DebugString() const {
  // Human-readable summary for quick logging
  std::ostringstream oss;
  oss << "BatchSpineMap{num_batches=" << num_batches_ << "}\n";
  for (int b = 0; b < num_batches_; ++b) {
    oss << "  Batch " << b << ":\n";
    for (int s = 0; s < kNumSpines; ++s) {
      const auto& v = batches_[b].phys_to_logic[s];
      oss << "    phys_spine[" << s << "]: size=" << v.size();
      if (!v.empty()) {
        oss << " addrs=[";
        for (size_t i = 0; i < v.size(); ++i) {
          oss << v[i];
          if (i + 1 < v.size()) oss << ", ";
        }
        oss << "]";
      }
      oss << "\n";
    }
  }
  return oss.str();
}

void BatchSpineMap::CheckBatch(int batchIdx) const {
  if (batchIdx < 0 || batchIdx >= num_batches_) {
    throw std::out_of_range("Batch index out of range");
  }
}

void BatchSpineMap::CheckSpine(int physSpineId) const {
  if (physSpineId < 0 || physSpineId >= kNumSpines) {
    throw std::out_of_range("Physical spine ID out of range");
  }
}

} // namespace driver
} // namespace sf
