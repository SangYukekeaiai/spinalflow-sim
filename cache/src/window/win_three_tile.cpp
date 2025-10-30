// All comments are in English.
#include "cache/window_iface.h"

#include <memory>
#include <array>

namespace sf::cache {

class ThreeTileWindow final : public IWindow {
public:
  void Set(int cur_tile, int next_tile) override {
    tiles_[0] = cur_tile;
    tiles_[1] = next_tile;
    tiles_[2] = (next_tile >= 0) ? next_tile + 1 : -1;
  }

  void Advance() override {
    tiles_[0] = tiles_[1];
    tiles_[1] = tiles_[2];
    tiles_[2] = (tiles_[1] >= 0) ? tiles_[1] + 1 : -1;
  }

  bool Allows(int tile_id) const override {
    return tiles_[0] == tile_id || tiles_[1] == tile_id || tiles_[2] == tile_id;
  }

  int Cur() const override { return tiles_[0]; }
  int Next() const override { return tiles_[1]; }

private:
  std::array<int, 3> tiles_{{-1, -1, -1}};
};

std::unique_ptr<IWindow> MakeThreeTileWindow() {
  return std::make_unique<ThreeTileWindow>();
}

} // namespace sf::cache

