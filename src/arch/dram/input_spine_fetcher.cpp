#include "arch/dram/input_spine_fetcher.hpp"

namespace sf { namespace dram {

InputSpineFetcher::InputSpineFetcher(const DramFormat&     fmt,
                                     const DramImage&      img,
                                     const LayerDirectory& dir,
                                     std::uint16_t         layer_id)
: fmt_(&fmt), img_(&img), dir_(&dir), layer_id_(layer_id) {
  // Nothing else to do; readers are created lazily per spine.
}

bool InputSpineFetcher::operator()(int /*batch_ignored*/,
                                   int spine,
                                   std::vector<std::uint8_t>& out_line,
                                   const sf::dram::DramFormat*& out_fmt)
{
  out_fmt = fmt_;
  out_line.clear();

  // Guard against negative inputs converted to huge uint16_t.
  if (spine < 0) return false;
  const std::uint16_t s = static_cast<std::uint16_t>(spine);

  // If we already know this spine is at EOF, fail fast.
  auto it_eof = eof_flags_.find(s);
  if (it_eof != eof_flags_.end() && it_eof->second) return false;

  // Ensure we have a reader bound to (layer_id_, spine).
  StreamReader* r = ensure_reader(s);
  if (!r) { eof_flags_[s] = true; return false; }

  // Try to read the next segment for this spine.
  SegmentHeader hdr{};
  if (!r->read_next(out_line, &hdr)) {
    // No more data for this spine; remember EOF so future calls are cheap.
    eof_flags_[s] = true;
    return false;
  }

  // A full line (header + payload + padding) is now in 'out_line'.
  // The caller (InputSpineBuffer) will parse it via the provided DramFormat.
  return true;
}

void InputSpineFetcher::SetLayer(std::uint16_t new_layer) {
  // Switch to a new layer: clear all per-spine readers and EOF states.
  layer_id_ = new_layer;
  readers_.clear();
  eof_flags_.clear();
}

void InputSpineFetcher::ResetAll() {
  readers_.clear();
  eof_flags_.clear();
}

bool InputSpineFetcher::Eof(std::uint16_t spine) const {
  auto it = eof_flags_.find(spine);
  return (it != eof_flags_.end()) ? it->second : false;
}

StreamReader* InputSpineFetcher::ensure_reader(std::uint16_t spine) {
  // Look up existing reader.
  auto it = readers_.find(spine);
  if (it != readers_.end()) return it->second.get();

  // Lazily construct a new reader bound to the shared fmt/img/dir.
  auto r = std::make_unique<StreamReader>(*fmt_, *img_, *dir_);
  if (!r->open_input(layer_id_, spine)) {
    // Inputs range may be empty or malformed for this layer; fail.
    return nullptr;
  }

  StreamReader* raw = r.get();
  readers_.emplace(spine, std::move(r));
  return raw;
}

}} // namespace sf::dram
