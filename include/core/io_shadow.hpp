#pragma once
// All comments are in English.

#include <cstdint>
#include <stdexcept>
#include <cmath>

namespace sf {

// Minimal compute-shadow credit: accumulate compute ticks,
// apply once to the next IO load, then reset to zero.
class IOShadow {
public:
  IOShadow() = default;
  explicit IOShadow(double bytes_per_cycle) : bpc_(bytes_per_cycle) {
    if (bpc_ <= 0.0) throw std::invalid_argument("IOShadow: bytes_per_cycle must be > 0.");
  }

  void SetBytesPerCycle(double bytes_per_cycle) {
    if (bytes_per_cycle <= 0.0) throw std::invalid_argument("IOShadow: bytes_per_cycle must be > 0.");
    bpc_ = bytes_per_cycle;
  }

  void OnComputeCycle(std::uint64_t cycles = 1) { credit_ += cycles; }

  std::uint64_t ApplyLoadBytes(std::uint64_t bytes) const {
    const std::uint64_t load_cycles = BytesToCycles(bytes);
    return ApplyLoadCycles(load_cycles);
  }

  // Apply current credit against the load and return blocking cycles.
  // After application, credit resets to 0 (one-shot semantics).
  std::uint64_t ApplyLoadCycles(std::uint64_t load_cycles) const {
    const std::uint64_t shadow = (credit_ >= load_cycles) ? load_cycles : credit_;
    const std::uint64_t block  = (load_cycles > shadow) ? (load_cycles - shadow) : 0ULL;
    return block;
  }

  // Reset credit to zero (caller should do this right after applying a load).
  void ResetCredit() { credit_ = 0; }

  std::uint64_t Credit() const { return credit_; }
  std::uint64_t BytesToCycles(std::uint64_t bytes) const {
    if (bytes == 0) return 0;
    if (bpc_ <= 0.0) throw std::logic_error("IOShadow: bytes_per_cycle not set");
    return static_cast<std::uint64_t>(std::llround(std::floor((bytes + bpc_ - 1.0) / bpc_)));
  }

private:
  double bpc_ = 16.0;           // bytes per cycle
  std::uint64_t credit_ = 0;    // accumulated compute cycles
};

} // namespace sf
