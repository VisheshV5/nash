#pragma once

#include <cstdint>

namespace regret {

inline std::uint64_t splitmix64(std::uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

// xoshiro256**: small, fast, good enough for Monte Carlo sampling.
class Rng {
 public:
  explicit Rng(std::uint64_t seed) {
    for (auto& w : s_) w = seed = splitmix64(seed);
  }

  // Independent stream for (seed, stream), e.g. one per CFR iteration.
  static Rng stream(std::uint64_t seed, std::uint64_t stream) {
    return Rng(splitmix64(seed) ^ splitmix64(stream + 0x632BE59BD9B4E019ULL));
  }

  std::uint64_t next() {
    const std::uint64_t result = rotl(s_[1] * 5, 7) * 9;
    const std::uint64_t t = s_[1] << 17;
    s_[2] ^= s_[0];
    s_[3] ^= s_[1];
    s_[1] ^= s_[2];
    s_[0] ^= s_[3];
    s_[2] ^= t;
    s_[3] = rotl(s_[3], 45);
    return result;
  }

  // Uniform in [0, n). The modulo bias is negligible for the small n used here.
  std::uint64_t below(std::uint64_t n) { return next() % n; }

  // Uniform in [0, 1).
  double uniform() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }

 private:
  static std::uint64_t rotl(std::uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
  std::uint64_t s_[4];
};

}  // namespace regret
