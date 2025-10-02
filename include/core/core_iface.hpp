#pragma once

namespace sf {

// All comments are in English.
class CoreIface {
public:
  virtual ~CoreIface() = default;

  // S0 -> S1 valid publishing/query
  virtual void SetSt0St1Valid(bool v) = 0;
  virtual bool st0_st1_valid() const = 0;

  // S1 -> S2 valid publishing/query
  virtual void SetSt1St2Valid(bool v) = 0;
  virtual bool st1_st2_valid() const = 0;

  // NEW: S2 -> S3 valid publishing/query
  virtual void SetSt2St3Valid(bool v) = 0;
  virtual bool st2_st3_valid() const = 0;

  // NEW: S3 -> S4 valid publishing/query
  virtual void SetSt3St4Valid(bool v) = 0;
  virtual bool st3_st4_valid() const = 0;

  // NEW: S4 -> S5 valid publishing/query
  virtual void SetSt4St5Valid(bool v) = 0;
  virtual bool st4_st5_valid() const = 0;
};

} // namespace sf
