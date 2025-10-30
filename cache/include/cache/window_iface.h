// All comments are in English.
#pragma once

namespace sf::cache {

class IWindow {
public:
  virtual ~IWindow() = default;
  virtual void Set(int cur_tile, int next_tile) = 0;
  virtual void Advance() = 0;
  virtual bool Allows(int tile_id) const = 0;
  virtual int Cur() const = 0;
  virtual int Next() const = 0;
};

} // namespace sf::cache

