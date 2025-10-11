// All comments are in English.

#include "arch/input_spine_buffer.hpp"

// Include your DRAM header (adjust path if needed in your repo).
#include "arch/dram/simple_dram.hpp"  // provides sf::dram::SimpleDRAM

namespace sf {

InputSpineBuffer::InputSpineBuffer(sf::dram::SimpleDRAM* dram)
  : num_phys_(kNumPhysISB),
    entries_per_buf_(kIsbEntries),
    bytes_per_buf_(static_cast<std::size_t>(kIsbEntries) * sizeof(Entry)),
    buffers_(static_cast<size_t>(kNumPhysISB)),
    read_idx_(static_cast<size_t>(kNumPhysISB), 0),
    valid_count_(static_cast<size_t>(kNumPhysISB), 0),
    logical_id_loaded_(static_cast<size_t>(kNumPhysISB), -1),
    dram_(dram)
{
  if (!dram_) {
    throw std::invalid_argument("InputSpineBuffer: null DRAM handle");
  }
  // Allocate per-physical-buffer storage.
  for (int i = 0; i < num_phys_; ++i) {
    buffers_[static_cast<size_t>(i)].resize(static_cast<size_t>(entries_per_buf_));
  }
}

void InputSpineBuffer::Reset() {
  std::fill(read_idx_.begin(), read_idx_.end(), 0);
  std::fill(valid_count_.begin(), valid_count_.end(), 0);
  std::fill(logical_id_loaded_.begin(), logical_id_loaded_.end(), -1);
}

bool InputSpineBuffer::PreloadFirstBatch(const std::vector<int>& logical_spine_ids_first_batch,
                                         int layer_id,
                                         uint64_t* out_cycles)
{
  if (logical_spine_ids_first_batch.empty()) {
    if (out_cycles) *out_cycles = 0;
    return false; // nothing to do
  }
  if (static_cast<int>(logical_spine_ids_first_batch.size()) > num_phys_) {
    throw std::invalid_argument("PreloadFirstBatch: more logical spines than physical buffers");
  }
  // Load into physical buffers and mark metadata.
  const uint64_t cycles = LoadBatchIntoBuffers_(logical_spine_ids_first_batch, layer_id);
  if (out_cycles) *out_cycles = cycles;
  return true;
}

bool InputSpineBuffer::run(const std::vector<int>& logical_spine_ids_current_batch,
                           int layer_id,
                           int current_batch_cursor,
                           int total_batches_needed,
                           uint64_t* out_cycles)
{
  if (out_cycles) *out_cycles = 0;
  // Guard: only attempt load when batches remain and all buffers are empty.
  if (current_batch_cursor < 0 || current_batch_cursor >= total_batches_needed) {
    return false; // no more batches to load or invalid cursor
  }
  if (!AllEmpty()) {
    std::cout << "InputSpineBuffer::run: buffers not empty, cannot load new batch yet.\n";
    return false; // not eligible to load; still draining current data
  }
  if (static_cast<int>(logical_spine_ids_current_batch.size()) > num_phys_) {
    throw std::invalid_argument("run(): more logical spines than physical buffers");
  }
  // Perform the load.
  const uint64_t cycles = LoadBatchIntoBuffers_(logical_spine_ids_current_batch, layer_id);
  if (out_cycles) *out_cycles = cycles;
  return true;
}

bool InputSpineBuffer::PopSmallestTsEntry(Entry& out) {
  int best_idx = -1;
  uint8_t best_ts = std::numeric_limits<uint8_t>::max();

  // Scan all physical buffers for the smallest head timestamp.
  for (int i = 0; i < num_phys_; ++i) {
    if (Available_(i) <= 0) continue;
    const Entry& head = buffers_[static_cast<size_t>(i)][static_cast<size_t>(read_idx_[static_cast<size_t>(i)])];
    if (best_idx < 0 || head.ts < best_ts) {
      best_idx = i;
      best_ts = head.ts;
    }
  }

  if (best_idx < 0) {
    return false; // all buffers empty
  }

  // Pop one entry from the chosen buffer.
  out = buffers_[static_cast<size_t>(best_idx)][static_cast<size_t>(read_idx_[static_cast<size_t>(best_idx)])];
  read_idx_[static_cast<size_t>(best_idx)] += 1;
  // When a buffer becomes fully consumed, we leave it as empty (no auto-reload here).
  // std::cout << "Popped Entry from buffer " << best_idx << ": (ts=" << static_cast<int>(out.ts)
  //           << ", neuron_id=" << out.neuron_id << ")\n";
  return true;
}

bool InputSpineBuffer::AllEmpty() const {
  for (int i = 0; i < num_phys_; ++i) {
    if (Available_(i) > 0) return false;
  }
  return true;
}

uint64_t InputSpineBuffer::LoadBatchIntoBuffers_(const std::vector<int>& logical_spine_ids,
                                             int layer_id)
{
  // Clear all physical buffers before loading the new batch.
  for (int i = 0; i < num_phys_; ++i) {
    read_idx_[static_cast<size_t>(i)] = 0;
    valid_count_[static_cast<size_t>(i)] = 0;
    logical_id_loaded_[static_cast<size_t>(i)] = -1;
  }

  uint64_t total_wire_bytes = 0;
  int num_loaded = 0;

  // Load each provided logical spine into the corresponding physical buffer slot.
  for (int i = 0; i < static_cast<int>(logical_spine_ids.size()); ++i) {
    const int spine_id = logical_spine_ids[static_cast<size_t>(i)];
    // Copy bytes directly into the physical buffer storage.
    const std::uint32_t copied_bytes = dram_->LoadInputSpine(
        static_cast<std::uint32_t>(layer_id),
        static_cast<std::uint32_t>(spine_id),
        static_cast<void*>(buffers_[static_cast<size_t>(i)].data()),
        static_cast<std::uint32_t>(bytes_per_buf_)
    );

    // Compute how many entries are valid (partial loads are allowed).
    const std::size_t entries = copied_bytes / sizeof(Entry);
    if (entries > static_cast<std::size_t>(entries_per_buf_)) {
      throw std::runtime_error("LoadBatchIntoBuffers_: DRAM returned more bytes than buffer capacity");
    }
    valid_count_[static_cast<size_t>(i)] = static_cast<int>(entries);
    read_idx_[static_cast<size_t>(i)] = 0;
    logical_id_loaded_[static_cast<size_t>(i)] = spine_id;

    total_wire_bytes += static_cast<uint64_t>(entries) * static_cast<uint64_t>(timing_.wire_entry_bytes);
    if (entries > 0) ++num_loaded;
  }

  const uint64_t denom_bw = static_cast<uint64_t>(std::max(1u, timing_.bw_bytes_per_cycle)) *
                            static_cast<uint64_t>(std::max(1u, timing_.parallel_loads));
  const uint64_t data_cycles  = CeilDivU64(total_wire_bytes, denom_bw);
  const uint64_t fixed_cycles = static_cast<uint64_t>(timing_.fixed_latency) *
                                CeilDivU64(static_cast<uint64_t>(num_loaded),
                                           static_cast<uint64_t>(std::max(1u, timing_.parallel_loads)));


  return data_cycles + fixed_cycles;
}

} // namespace sf
