#pragma once
#include <array>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <optional>
#include "common/output_token.hpp"

namespace sf {

// Fixed PE count; switch to common/constants.hpp if available in your repo.
#ifndef SF_K_NUM_PES_DEFINED
static constexpr int kNumPEs = 128;
#endif

// A tiny bounded FIFO storing produced PE indices for the current step.
// This lets the serializer iterate only the PEs that actually produced an output,
// in O(num_outputs) rather than O(kNumPEs).
class IndexFIFO {
public:
    explicit IndexFIFO(std::size_t capacity = kNumPEs)
      : buf_(capacity), head_(0), tail_(0), size_(0) {}

    void Clear() {
        head_ = tail_ = size_ = 0;
    }

    bool Push(int pe_idx) {
        if (Full()) return false;
        buf_[tail_] = pe_idx;
        tail_ = (tail_ + 1) % buf_.size();
        ++size_;
        return true;
    }

    bool Pop(int& out) {
        if (Empty()) return false;
        out = buf_[head_];
        head_ = (head_ + 1) % buf_.size();
        --size_;
        return true;
    }

    bool Front(int& out) const {
        if (Empty()) return false;
        out = buf_[head_];
        return true;
    }

    bool Empty() const { return size_ == 0; }
    bool Full()  const { return size_ == buf_.size(); }
    std::size_t Size() const { return size_; }
    std::size_t Capacity() const { return buf_.size(); }

    // Iterate without popping (read-only).
    template <typename F>
    void ForEach(F&& f) const {
        std::size_t idx = head_;
        for (std::size_t n = 0; n < size_; ++n) {
            f(buf_[idx]);
            idx = (idx + 1) % buf_.size();
        }
    }

private:
    std::vector<int> buf_;
    std::size_t head_, tail_, size_;
};

// OutputHarvest captures at most one token per PE per step.
// It keeps a presence mask, a per-PE slot, and an IndexFIFO of which PEs produced.
class OutputHarvest {
public:
    OutputHarvest();

    // Remove all captured tokens and reset state for next step.
    void Clear();

    // Capture one token for a specific PE. Returns false if already captured this step.
    bool Capture(int pe_id, const OutputToken& tok);

    // Query helpers
    bool Has(int pe_id) const;
    const OutputToken& Get(int pe_id) const;
    int  Count() const { return static_cast<int>(count_); }

    // Iterate tokens in the order they were captured (IndexFIFO order).
    template <typename F>
    void ForEach(F&& f) const {
        index_fifo_.ForEach([&](int pe){
            f(pe, slots_[pe]);
        });
    }

    // Random access iteration (avoid O(N) scan) by exposing the index FIFO.
    const IndexFIFO& ProducedIndexFIFO() const { return index_fifo_; }

private:
    std::array<bool, kNumPEs>         produced_{};
    std::array<OutputToken, kNumPEs>  slots_{};
    IndexFIFO                         index_fifo_;
    std::size_t                       count_ = 0;
};

} // namespace sf
