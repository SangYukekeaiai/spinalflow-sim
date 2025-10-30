// All comments are in English.
#include "cache/window_iface.h"

#include <memory>

namespace sf::cache {

class TwoTileWindow final : public IWindow {
public:
  void Set(int cur_tile, int next_tile) override {
    cur_tile_ = cur_tile;
    next_tile_ = next_tile;
  }

  void Advance() override {
    cur_tile_ = next_tile_;
    if (next_tile_ >= 0) {
      next_tile_ = next_tile_ + 1;
    }
  }

  bool Allows(int tile_id) const override {
    return tile_id == cur_tile_ || tile_id == next_tile_;
  }

  int Cur() const override { return cur_tile_; }
  int Next() const override { return next_tile_; }

private:
  int cur_tile_ = -1;
  int next_tile_ = -1;
};

std::unique_ptr<IWindow> MakeTwoTileWindow() {
  return std::make_unique<TwoTileWindow>();
}

} // namespace sf::cache

