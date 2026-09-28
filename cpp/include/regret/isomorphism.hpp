#pragma once

#include <array>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "regret/cards.hpp"

namespace regret {

// Maps a deal split into rounds (e.g. {2, 3} = hole cards + flop) to a dense index in
// [0, size()), such that two deals share an index exactly when one is a suit permutation of
// the other. Order within a round doesn't matter; order across rounds does.
//
// Sizes for hold'em: {2} -> 169, {2,3} -> 1,286,792, {2,3,1} -> 13,960,050,
// {2,3,1,1} -> 123,156,254.
//
// How it works: two deals are isomorphic iff they have the same multiset of per-suit
// "patterns", where a suit's pattern is the rank set it holds in each round. So:
//   1. a suit's *shape* is its card count per round, and its *key* is a dense index of its
//      rank sets given the shape;
//   2. a *config* is the multiset of the 4 suit shapes. Each config owns a contiguous block
//      of indices;
//   3. inside a config, suits with equal shapes form groups, and each group contributes a
//      multiset-of-keys index (mixed radix across groups).
class HandIndexer {
 public:
  static constexpr int kMaxRounds = 4;
  static constexpr int kMaxCardsPerRound = 7;

  explicit HandIndexer(std::vector<int> cards_per_round);

  std::uint64_t size() const { return size_; }
  int rounds() const { return static_cast<int>(cards_per_round_.size()); }
  int cards_in_round(int round) const { return cards_per_round_[round]; }
  int total_cards() const { return total_cards_; }

  // `cards` holds total_cards() distinct cards, round by round.
  std::uint64_t index(const Card* cards) const;

  // Writes the canonical representative of `idx` to `out` (total_cards() cards, round by round,
  // ascending within each round). index(out) == idx.
  void unindex(std::uint64_t idx, Card* out) const;

 private:
  using Shape = std::array<std::uint8_t, kMaxRounds>;

  struct Group {
    Shape shape;
    int count;             // number of suits with this shape
    std::uint64_t keys;    // key space size for one suit of this shape
    std::uint64_t combos;  // multisets of `count` keys
  };

  struct Config {
    std::uint64_t offset;
    std::uint64_t size;
    std::vector<Group> groups;  // in canonical suit order
  };

  std::uint32_t pack(const Shape& s) const;
  std::uint64_t key_space(const Shape& s) const;
  std::uint64_t suit_key(const Shape& s, const std::array<std::uint32_t, kMaxRounds>& sets) const;
  void suit_sets(const Shape& s, std::uint64_t key,
                 std::array<std::uint32_t, kMaxRounds>& sets) const;
  void enumerate_configs();

  std::vector<int> cards_per_round_;
  int total_cards_ = 0;
  std::uint64_t size_ = 0;
  std::vector<Config> configs_;
  std::unordered_map<std::uint64_t, int> config_by_shapes_;
};

}  // namespace regret
