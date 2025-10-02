#pragma once
#include <cstdint>
#include <vector>
#include <stdexcept>

namespace sf { namespace dram {

// All comments are in English.

struct Range {
  std::uint64_t begin = 0;  // inclusive
  std::uint64_t end   = 0;  // exclusive
  bool empty() const { return begin >= end; }
  std::uint64_t size() const { return (end > begin) ? (end - begin) : 0; }
};

struct LayerEntry {
  Range inputs;   // all input spine segments for this layer (batch-agnostic)
  Range weights;  // all weight segments for this layer
  Range outputs;  // all output spine segments for this layer
};

class LayerDirectory {
public:
  void reset(int num_layers) {
    if (num_layers < 0) throw std::invalid_argument("LayerDirectory::reset: negative layers");
    layers_.assign(static_cast<std::size_t>(num_layers), LayerEntry{});
  }

  int num_layers() const { return static_cast<int>(layers_.size()); }

  void set_input_range(int L, Range r) {
    check_layer(L);
    layers_[static_cast<std::size_t>(L)].inputs = r;
  }
  const Range& input_range(int L) const {
    check_layer(L);
    return layers_[static_cast<std::size_t>(L)].inputs;
  }

  void set_weights_range(int L, Range r) {
    check_layer(L);
    layers_[static_cast<std::size_t>(L)].weights = r;
  }
  const Range& weights_range(int L) const {
    check_layer(L);
    return layers_[static_cast<std::size_t>(L)].weights;
  }

  void set_output_range(int L, Range r) {
    check_layer(L);
    layers_[static_cast<std::size_t>(L)].outputs = r;
  }
  const Range& output_range(int L) const {
    check_layer(L);
    return layers_[static_cast<std::size_t>(L)].outputs;
  }

private:
  void check_layer(int L) const {
    if (L < 0 || L >= num_layers()) {
      throw std::out_of_range("LayerDirectory: layer index");
    }
  }

private:
  std::vector<LayerEntry> layers_;
};

}} // namespace sf::dram
